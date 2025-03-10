import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import threading
import skvideo.io
from queue import Queue, Empty
import shutil
import tempfile
import sys

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field

# Import the pytorch_msssim module from RIFE
sys.path.append('ECCV2022-RIFE')
from model.pytorch_msssim import ssim_matlab

# Reimplement transferAudio function instead of importing it
def transfer_audio(source_video, target_video):
    """Transfer audio from source video to target video."""
    temp_dir = tempfile.mkdtemp()
    temp_audio = os.path.join(temp_dir, "audio.mkv")
    
    # Extract audio from original video
    os.system(f'ffmpeg -y -i "{source_video}" -c:a copy -vn {temp_audio}')
    
    # Rename the target video (to add audio later)
    target_no_audio = os.path.splitext(target_video)[0] + "_noaudio" + os.path.splitext(target_video)[1]
    os.rename(target_video, target_no_audio)
    
    # Merge audio with the new video
    os.system(f'ffmpeg -y -i "{target_no_audio}" -i {temp_audio} -c copy "{target_video}"')
    
    # Check if the merge was successful
    if os.path.getsize(target_video) == 0:
        # Try with AAC audio if the lossless transfer failed
        temp_audio = os.path.join(temp_dir, "audio.m4a")
        os.system(f'ffmpeg -y -i "{source_video}" -c:a aac -b:a 160k -vn {temp_audio}')
        os.system(f'ffmpeg -y -i "{target_no_audio}" -i {temp_audio} -c copy "{target_video}"')
        
        if os.path.getsize(target_video) == 0:
            # If still failed, use the no-audio version
            os.rename(target_no_audio, target_video)
            print("Audio transfer failed. Output video will have no audio.")
        else:
            # Remove the no-audio version if succeeded
            os.remove(target_no_audio)
            print("Audio was transcoded to AAC format.")
    else:
        # Remove the no-audio version if succeeded
        os.remove(target_no_audio)
        print("Audio transfer successful.")
    
    # Clean up
    shutil.rmtree(temp_dir)

class AppInput(BaseAppInput):
    """Input for the RIFE video interpolation app."""
    video: File = Field(
        description="Input video file to be processed for frame interpolation"
    )
    exp: int = Field(
        default=1,
        description="Interpolation factor: 2^exp frames will be generated between each original frame",
        ge=0,
        le=4
    )
    scale: float = Field(
        default=1.0, 
        description="Scale factor for processing. Lower values (0.5, 0.25) recommended for 4K videos",
        enum=[0.25, 0.5, 1.0, 2.0, 4.0]
    )
    fps: int = Field(
        default=None, 
        description="Target FPS. If None, will use source_fps * 2^exp"
    )
    keep_audio: bool = Field(
        default=True, 
        description="Whether to transfer audio from source to output video"
    )

class AppOutput(BaseAppOutput):
    """Output for the RIFE video interpolation app."""
    interpolated_video: File

class App(BaseApp):
    async def setup(self):
        """Initialize the RIFE model and resources."""
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        warnings.filterwarnings("ignore")
        
        # Load model using the same approach as in inference_video.py
        try:
            try:
                try:
                    from model.RIFE_HDv2 import Model
                    self.model = Model()
                    self.model.load_model('ECCV2022-RIFE/train_log', -1)
                    print("Loaded v2.x HD model.")
                except:
                    from train_log.RIFE_HDv3 import Model
                    self.model = Model()
                    self.model.load_model('ECCV2022-RIFE/train_log', -1)
                    print("Loaded v3.x HD model.")
            except:
                from model.RIFE_HD import Model
                self.model = Model()
                self.model.load_model('ECCV2022-RIFE/train_log', -1)
                print("Loaded v1.x HD model")
        except:
            from model.RIFE import Model
            self.model = Model()
            self.model.load_model('ECCV2022-RIFE/train_log', -1)
            print("Loaded ArXiv-RIFE model")
        
        self.model.eval()
        self.model.device()
        self.ssim_func = ssim_matlab
    
    def make_inference(self, I0, I1, n, scale):
        """Generate n intermediate frames between I0 and I1."""
        middle = self.model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = self.make_inference(I0, middle, n=n//2, scale=scale)
        second_half = self.make_inference(middle, I1, n=n//2, scale=scale)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run video interpolation on the input video."""
        # Create temporary directory and files
        input_path = input_data.video.path
        
        # Use NamedTemporaryFile for the output file
        output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_file.close()  # Close file handle but keep the file
        output_path = output_file.name
        
        # Read video info
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Set output FPS
        output_fps = input_data.fps if input_data.fps is not None else fps * (2 ** input_data.exp)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Calculate padding values for model
        tmp = max(32, int(32 / input_data.scale))
        ph = ((height - 1) // tmp + 1) * tmp
        pw = ((width - 1) // tmp + 1) * tmp
        padding = (0, pw - width, 0, ph - height)
        
        # Initialize read/write buffers and threads
        write_buffer = Queue(maxsize=500)
        read_buffer = Queue(maxsize=500)
        
        # Function to read frames from video
        def build_read_buffer(read_buffer, video_path):
            videogen = skvideo.io.vreader(video_path)
            try:
                for frame in videogen:
                    read_buffer.put(frame)
            except Exception as e:
                print(f"Error reading video: {e}")
            read_buffer.put(None)  # Signal end of video
        
        # Function to write frames to output video
        def clear_write_buffer(write_buffer, vid_out):
            frame_count = 0
            while True:
                item = write_buffer.get()
                if item is None:
                    break
                vid_out.write(item[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                frame_count += 1
            return frame_count
        
        # Start reader thread
        reader_thread = threading.Thread(
            target=build_read_buffer,
            args=(read_buffer, input_path)
        )
        reader_thread.start()
        
        # Start writer thread and get frame count
        writer_thread = threading.Thread(
            target=clear_write_buffer,
            args=(write_buffer, vid_out)
        )
        writer_thread.start()
        
        # Read first frame
        lastframe = read_buffer.get()
        if lastframe is None:
            raise ValueError("Input video is empty or could not be read")
        
        # Process video frames
        I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = F.pad(I1, padding)
        temp = None  # To save frames when processing static frames
        
        print(f"Processing video with {total_frames} frames, exp={input_data.exp}, scale={input_data.scale}")
        
        # Process each frame in the video
        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = read_buffer.get()
            
            if frame is None:
                break
                
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = F.pad(I1, padding)
            
            # Check frame similarity to detect static scenes
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = self.ssim_func(I0_small[:, :3], I1_small[:, :3])
            
            break_flag = False
            if ssim > 0.996:  # Almost identical frame, potential static scene
                frame = read_buffer.get()
                if frame is None:
                    break_flag = True
                    frame = lastframe
                else:
                    temp = frame
                I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
                I1 = F.pad(I1, padding)
                I1 = self.model.inference(I0, I1, input_data.scale)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = self.ssim_func(I0_small[:, :3], I1_small[:, :3])
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:height, :width]
            
            # Generate intermediate frames
            if ssim < 0.2:  # Scene change, simple frame duplication
                output = []
                for i in range((2 ** input_data.exp) - 1):
                    output.append(I0)
            else:
                output = self.make_inference(I0, I1, 2**input_data.exp-1, input_data.scale) if input_data.exp else []
            
            # Write frames to output
            write_buffer.put(lastframe)
            
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:height, :width])
                
            lastframe = frame
            if break_flag:
                break
        
        # Write the last frame
        write_buffer.put(lastframe)
        write_buffer.put(None)  # Signal end of writing
        
        # Wait for writer thread to finish
        writer_thread.join()
        
        # Close resources
        vid_out.release()
        reader_thread.join()
        
        # Transfer audio if requested
        if input_data.keep_audio:
            try:
                transfer_audio(input_path, output_path)
            except Exception as e:
                print(f"Audio transfer failed: {e}")
        
        print("Video interpolation complete")
        
        return AppOutput(interpolated_video=File(path=output_path))

    async def unload(self):
        """Clean up resources."""
        # Clear model from memory
        if hasattr(self, 'model'):
            del self.model
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()