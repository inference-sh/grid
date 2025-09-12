import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import skvideo.io
import tempfile
import time

# Add current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add ECCV2022-RIFE directory to Python path
rife_dir = os.path.join(current_dir, 'ECCV2022-RIFE')
if rife_dir not in sys.path:
    sys.path.append(rife_dir)

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional

warnings.filterwarnings("ignore")

class AppInput(BaseAppInput):
    video: File = Field(description="Input video file to interpolate")
    exp: int = Field(default=1, ge=1, le=3, description="Interpolation level (1=2x, 2=4x, 3=8x frames)")
    scale: float = Field(default=1.0, description="Scale factor for processing (0.5 for 4K video, 1.0 for HD)")
    fps: Optional[int] = Field(default=None, description="Target output fps (if None, auto-calculated from input fps * 2^exp)")
    fp16: bool = Field(default=True, description="Use fp16 mode for faster inference")
    uhd: bool = Field(default=False, description="Enable 4K video support (automatically sets scale=0.5)")

class AppOutput(BaseAppOutput):
    video: File = Field(description="Output video with interpolated frames")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize RIFE model and resources."""
        # Use explicit device detection like original RIFE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Disable gradients for inference
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        # Load RIFE model with fallback strategy
        self.model = self._load_rife_model()
        self.model.eval()
        self.model.device()
        
        print(f"RIFE model loaded successfully on device: {self.device}")
    
    def _load_rife_model(self):
        """Load RIFE model with fallback strategy similar to original inference_video.py"""
        model_dir = "ECCV2022-RIFE/train_log"
        
        try:
            try:
                try:
                    from model.oldmodel.RIFE_HDv2 import Model
                    model = Model()
                    model.load_model(model_dir, -1)
                    print("Loaded v2.x HD model.")
                    return model
                except Exception as e:
                    print(f"Failed to load HDv2: {e}")
                    from train_log.RIFE_HDv3 import Model
                    model = Model()
                    model.load_model(model_dir, -1)
                    print("Loaded v3.x HD model.")
                    return model
            except Exception as e:
                print(f"Failed to load HDv3: {e}")
                from model.oldmodel.RIFE_HD import Model
                model = Model()
                model.load_model(model_dir, -1)
                print("Loaded v1.x HD model")
                return model
        except Exception as e:
            print(f"Failed to load HD model: {e}")
            from model.RIFE import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded ArXiv-RIFE model")
            return model

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run RIFE video interpolation on the input video."""
        # Validate input file
        if not input_data.video.exists():
            raise RuntimeError(f"Input video does not exist at path: {input_data.video.path}")
        
        # Configure parameters
        scale = 0.5 if input_data.uhd and input_data.scale == 1.0 else input_data.scale
        assert scale in [0.25, 0.5, 1.0, 2.0, 4.0], "Scale must be one of [0.25, 0.5, 1.0, 2.0, 4.0]"
        
        # Handle fp16 mode
        use_fp16 = input_data.fp16 and torch.cuda.is_available()
        if use_fp16:
            print("Using fp16 mode")
            # Note: We don't convert the RIFE model to half precision as it doesn't support it
            # Instead, we handle tensor dtype conversion explicitly
        
        # Get video properties
        videoCapture = cv2.VideoCapture(input_data.video.path)
        original_fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        videoCapture.release()
        
        # Calculate output fps
        if input_data.fps is None:
            output_fps = original_fps * (2 ** input_data.exp)
        else:
            output_fps = float(input_data.fps)
        
        # Create temporary output files
        temp_output_fd, temp_output_path = tempfile.mkstemp(suffix='_temp.mp4')
        os.close(temp_output_fd)
        final_output_fd, final_output_path = tempfile.mkstemp(suffix='.mp4')
        os.close(final_output_fd)
        
        try:
            # Process video to temporary file
            total_interpolated_frames = self._interpolate_video(
                input_data.video.path,
                temp_output_path,
                scale=scale,
                exp=input_data.exp,
                output_fps=output_fps,
                fp16=use_fp16,
                tot_frame=tot_frame
            )
            
            # Re-encode to browser-friendly format with ffmpeg
            self._reencode_for_browser(temp_output_path, final_output_path, output_fps)
            
            return AppOutput(
                video=File(path=final_output_path)
            )
        except Exception as e:
            # Clean up temporary files on error
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            raise e
        finally:
            # Clean up intermediate temp file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
    
    def _interpolate_video(self, input_path, output_path, scale, exp, output_fps, fp16, tot_frame):
        """Core video interpolation logic adapted from inference_video.py"""
        print(f"Starting video interpolation: {input_path} -> {output_path}")
        print(f"Parameters: scale={scale}, exp={exp}, output_fps={output_fps}, fp16={fp16}")
        
        # Setup video reader and writer
        videogen = skvideo.io.vreader(input_path)
        lastframe = next(videogen)
        h, w, _ = lastframe.shape
        print(f"Video dimensions: {w}x{h}")
        
        # Use H.264 codec for better browser compatibility
        # Try different fourcc codes in order of preference
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264 (best browser compatibility)
            cv2.VideoWriter_fourcc(*'h264'),  # H.264 alternative
            cv2.VideoWriter_fourcc(*'x264'),  # x264 encoder
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # Fallback to original
        ]
        
        vid_out = None
        for fourcc in fourcc_options:
            vid_out = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))
            if vid_out.isOpened():
                print(f"Using video codec: {fourcc}")
                break
            vid_out.release()
            vid_out = None
        
        if vid_out is None:
            raise RuntimeError("Failed to initialize video writer with any codec")
        
        # Import SSIM function from RIFE
        from model.pytorch_msssim import ssim_matlab
        
        # Setup padding for processing
        tmp = max(32, int(32 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        print(f"Padding: {padding}")
        
        def pad_image(img):
            # Always return float32 for RIFE model compatibility
            return F.pad(img, padding)
        
        def make_inference(I0, I1, n):
            """Recursive interpolation function"""
            middle = self.model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = make_inference(I0, middle, n=n//2)
            second_half = make_inference(middle, I1, n=n//2)
            if n%2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]
        
        # Read all frames into memory (simpler approach)
        all_frames = [lastframe]
        frame_count = 1
        
        print("Reading all frames...")
        try:
            for frame in videogen:
                all_frames.append(frame)
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Read {frame_count} frames...")
        except Exception as e:
            print(f"Finished reading frames at {frame_count}: {e}")
        
        print(f"Total input frames: {len(all_frames)}")
        
        # Initialize progress bar 
        pbar = tqdm(total=len(all_frames)-1, desc="Interpolating frames")
        
        # Process first frame - always use float32 for RIFE model compatibility
        I1 = torch.from_numpy(np.transpose(all_frames[0], (2,0,1))).to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0) / 255.
        I1 = pad_image(I1)
        
        # Write first frame
        vid_out.write(all_frames[0][:, :, ::-1])
        output_frame_count = 1
        
        # Main processing loop - process pairs of consecutive frames
        for i in range(1, len(all_frames)):
            I0 = I1  # Previous frame
            current_frame = all_frames[i]
            # Always use float32 for RIFE model compatibility
            I1 = torch.from_numpy(np.transpose(current_frame, (2,0,1))).to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0) / 255.
            I1 = pad_image(I1)
            
            # Check similarity using SSIM to avoid interpolating static frames
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            # Convert to float32 for SSIM calculation to avoid dtype mismatch in fp16 mode
            ssim = ssim_matlab(I0_small[:, :3].float(), I1_small[:, :3].float())
            
            if ssim < 0.2:  # Very different frames, use simple duplication
                print(f"Frame {i}: Very different frames (SSIM={ssim:.3f}), duplicating")
                # Just duplicate the first frame for the interpolated frames
                for _ in range((2 ** exp) - 1):
                    vid_out.write(all_frames[i-1][:, :, ::-1])
                    output_frame_count += 1
            elif ssim > 0.996:  # Very similar frames, skip interpolation
                print(f"Frame {i}: Very similar frames (SSIM={ssim:.3f}), skipping interpolation")
                # Skip interpolation but still write some frames
                for _ in range((2 ** exp) - 1):
                    vid_out.write(all_frames[i-1][:, :, ::-1])
                    output_frame_count += 1
            else:
                # Generate interpolated frames
                print(f"Frame {i}: Interpolating (SSIM={ssim:.3f})")
                interpolated = make_inference(I0, I1, 2**exp-1) if exp > 0 else []
                
                for mid in interpolated:
                    # Convert tensor back to numpy array
                    mid_frame = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                    vid_out.write(mid_frame[:, :, ::-1])  # BGR for OpenCV
                    output_frame_count += 1
            
            # Write the current frame
            vid_out.write(current_frame[:, :, ::-1])
            output_frame_count += 1
            
            pbar.update(1)
        
        pbar.close()
        vid_out.release()
        
        print(f"Interpolation complete. Output frames: {output_frame_count}")
        return output_frame_count
    
    def _reencode_for_browser(self, input_path, output_path, fps):
        """Re-encode video to browser-friendly H.264 format using ffmpeg with fallback"""
        import subprocess
        import shutil
        
        print(f"Re-encoding video for browser compatibility: {input_path} -> {output_path}")
        
        # Check if ffmpeg is available
        if shutil.which('ffmpeg') is None:
            print("ffmpeg not found, copying file without re-encoding")
            shutil.copy2(input_path, output_path)
            return
            
        # ffmpeg command for browser-friendly H.264 encoding
        # -c:v libx264: Use H.264 codec
        # -preset medium: Balance between speed and compression
        # -crf 18: High quality (lower values = higher quality, 18 is visually lossless)
        # -pix_fmt yuv420p: Pixel format compatible with most browsers/players
        # -movflags +faststart: Optimize for web streaming (moov atom at beginning)
        # -profile:v high: Use high profile for better compression (changed from baseline)
        # -level 4.0: H.264 level for better compatibility (changed from 3.1)
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-i', input_path,  # Input file
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'medium',  # Encoding speed/quality balance
            '-crf', '18',  # Constant Rate Factor (high quality)
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-profile:v', 'high',  # H.264 profile for better compression
            '-level', '4.0',  # H.264 level
            '-movflags', '+faststart',  # Optimize for web streaming
            '-r', str(fps),  # Frame rate
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            '-vsync', '0',  # Maintain original timestamps
            '-start_at_zero',  # Start timestamps at zero
            '-fflags', '+genpts',  # Generate presentation timestamps
            output_path
        ]
        
        try:
            print(f"Running ffmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)  # 5 minute timeout
            print("Re-encoding completed successfully")
            
            # Check if output file exists and has reasonable size
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Re-encoded video file is empty or doesn't exist")
                
        except subprocess.TimeoutExpired:
            print("ffmpeg re-encoding timed out, using original file")
            shutil.copy2(input_path, output_path)
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error (stdout): {e.stdout}")
            print(f"ffmpeg error (stderr): {e.stderr}")
            print("ffmpeg re-encoding failed, using original file")
            shutil.copy2(input_path, output_path)
        except Exception as e:
            print(f"Re-encoding failed with error: {e}, using original file")
            shutil.copy2(input_path, output_path)

    async def unload(self):
        """Clean up resources here."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()