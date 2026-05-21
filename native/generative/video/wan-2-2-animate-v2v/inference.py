import os
import sys
import tempfile
import torch
import numpy as np
import cv2
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional

# Add Wan2.2 directory to Python path for local imports
wan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wan2.2")
sys.path.insert(0, wan_dir)

from wan.animate import WanAnimate
from wan.configs.wan_animate_14B import animate_14B as config

class AppInput(BaseAppInput):
    # Raw input - no preprocessing required!
    driving_video: File = Field(description="Driving video - person performing the actions you want to transfer")
    reference_image: File = Field(description="Reference character image - the character you want to animate")

    # Preprocessing options
    mode: str = Field(default="animation", description="Mode: 'animation' (character animation) or 'replacement' (replace character in video)")
    resolution_width: int = Field(default=1280, ge=512, le=1920, description="Target width for processing")
    resolution_height: int = Field(default=720, ge=512, le=1920, description="Target height for processing")
    fps: int = Field(default=30, ge=-1, le=60, description="Target FPS (-1 to use original video FPS)")

    # Preprocessing - animation mode
    retarget_flag: bool = Field(default=False, description="Enable pose retargeting (recommended for different body proportions)")
    use_flux: bool = Field(default=False, description="Use FLUX image editing for better pose retargeting (slower but better quality)")

    # Preprocessing - replacement mode
    mask_iterations: int = Field(default=3, ge=1, le=10, description="Mask dilation iterations (replacement mode)")
    mask_kernel_size: int = Field(default=7, ge=3, le=15, description="Mask dilation kernel size (replacement mode)")
    mask_w_subdivisions: int = Field(default=1, ge=1, le=5, description="Mask width subdivisions for detail (replacement mode)")
    mask_h_subdivisions: int = Field(default=1, ge=1, le=5, description="Mask height subdivisions for detail (replacement mode)")

    # Generation parameters
    clip_len: int = Field(default=77, ge=5, le=121, description="Frames per clip (must be 4n+1, e.g., 77, 81, 121)")
    refert_num: int = Field(default=1, description="Temporal guidance frames (1 or 5 recommended)")
    shift: float = Field(default=5.0, ge=1.0, le=10.0, description="Noise schedule shift parameter")
    sample_solver: str = Field(default="dpm++", description="Sampling solver (dpm++ or unipc)")
    sampling_steps: int = Field(default=20, ge=10, le=50, description="Number of diffusion steps")
    guide_scale: float = Field(default=1.0, ge=1.0, le=10.0, description="Classifier-free guidance scale for expression control")

    # Prompts
    input_prompt: str = Field(default="", description="Text prompt (leave empty for default: '视频中的人在做动作')")
    n_prompt: str = Field(default="", description="Negative prompt")

    # Seed
    seed: int = Field(default=-1, description="Random seed (-1 for random)")

    # Model offloading
    offload_model: bool = Field(default=True, description="Offload models to CPU to save VRAM")

class AppOutput(BaseAppOutput):
    output_video: File = Field(description="Generated animation video")

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize WanAnimate model and preprocessing pipeline."""
        from accelerate import Accelerator
        from huggingface_hub import snapshot_download

        # Enable fast HuggingFace transfers
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # Initialize accelerator for device management
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Get device_id as integer for WanAnimate
        if hasattr(self.device, 'index') and self.device.index is not None:
            self.device_id = self.device.index
        else:
            self.device_id = 0 if str(self.device) == 'cuda' else 'cpu'

        print("📥 Downloading WAN Animate model checkpoints from HuggingFace...")
        checkpoint_dir = snapshot_download(
            repo_id="Wan-AI/Wan2.2-Animate-14B",
            resume_download=True,
            local_files_only=False,
        )
        print(f"✅ Model checkpoints downloaded to: {checkpoint_dir}")

        # Download preprocessing checkpoints
        print("📥 Downloading preprocessing model checkpoints...")
        self.preprocess_ckpt_dir = snapshot_download(
            repo_id="Wan-AI/Wan2.2-Animate-14B",
            allow_patterns=["process_checkpoint/**"],
            resume_download=True,
            local_files_only=False,
        )
        self.preprocess_ckpt_dir = os.path.join(self.preprocess_ckpt_dir, "process_checkpoint")
        print(f"✅ Preprocessing checkpoints downloaded to: {self.preprocess_ckpt_dir}")

        # Initialize preprocessing pipeline
        print("🔧 Initializing preprocessing pipeline...")
        self._init_preprocessing()
        print("✅ Preprocessing pipeline initialized")

        # Initialize WanAnimate model
        print("🔧 Initializing WanAnimate model...")
        self.wan_animate = WanAnimate(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device_id=self.device_id if isinstance(self.device_id, int) else 0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=True,
            convert_model_dtype=False,
            use_relighting_lora=False
        )
        print("✅ WanAnimate model initialized successfully")

        # Patch the generate method to add aggressive memory offloading
        self._patch_generate_for_memory_efficiency()

    def _patch_generate_for_memory_efficiency(self):
        """Patch WanAnimate.generate to add aggressive model offloading for memory efficiency.

        Pipeline flow:
        1. Text Encoder (T5) -> encode prompts -> offload
        2. VAE Encoder -> encode images/videos -> offload
        3. CLIP -> encode reference image -> offload
        4. Transformer (noise_model) -> diffusion sampling -> offload
        5. VAE Decoder -> decode latents to video -> offload
        """
        import gc
        import torch

        def cleanup_memory():
            """Clear CUDA cache and run garbage collection."""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        # 1. Wrap Text Encoder to offload after use
        # Already handled by original code with offload_model flag

        # 2. Wrap VAE encoder to offload after initial encoding phase
        original_vae_encode = self.wan_animate.vae.encode
        vae_encode_calls = [0]  # Track calls to know when encoding phase is done

        def offloading_vae_encode(*args, **kwargs):
            result = original_vae_encode(*args, **kwargs)
            vae_encode_calls[0] += 1
            # After ~5 encode calls, the initial setup is done, offload VAE encoder
            if vae_encode_calls[0] == 5:
                print("🔧 Offloading VAE encoder to CPU after initial encoding...")
                # Move encoder to CPU but keep decoder for later
                if hasattr(self.wan_animate.vae, 'encoder'):
                    self.wan_animate.vae.encoder.cpu()
                cleanup_memory()
            return result

        self.wan_animate.vae.encode = offloading_vae_encode

        # 3. Wrap CLIP visual encoder to offload after first use
        original_clip_visual = self.wan_animate.clip.visual
        clip_used = [False]

        def offloading_clip_visual(*args, **kwargs):
            result = original_clip_visual(*args, **kwargs)
            if not clip_used[0]:
                clip_used[0] = True
                print("🔧 Offloading CLIP to CPU after encoding reference...")
                self.wan_animate.clip.model.cpu()
                cleanup_memory()
            return result

        self.wan_animate.clip.visual = offloading_clip_visual

        # 4. Wrap transformer's forward to offload after sampling completes
        # We'll wrap the VAE decode instead since it's called after sampling
        original_vae_decode = self.wan_animate.vae.decode
        transformer_offloaded = [False]

        def offloading_vae_decode(*args, **kwargs):
            # Offload transformer before VAE decode (sampling is complete at this point)
            if not transformer_offloaded[0] and hasattr(self.wan_animate, 'noise_model'):
                transformer_offloaded[0] = True
                print("🔧 Offloading transformer to CPU before VAE decode...")
                self.wan_animate.noise_model.cpu()

                # Also bring VAE encoder back to GPU if needed for decoding
                if hasattr(self.wan_animate.vae, 'encoder'):
                    self.wan_animate.vae.encoder.to(self.accelerator.device)

                cleanup_memory()

            result = original_vae_decode(*args, **kwargs)
            return result

        self.wan_animate.vae.decode = offloading_vae_decode

        # 5. After decode completes, offload VAE decoder too
        def final_offloading_vae_decode(*args, **kwargs):
            result = offloading_vae_decode(*args, **kwargs)

            # After decoding, offload entire VAE
            print("🔧 Offloading VAE to CPU after decode...")
            self.wan_animate.vae.encoder.cpu()
            self.wan_animate.vae.decoder.cpu()
            cleanup_memory()

            return result

        self.wan_animate.vae.decode = final_offloading_vae_decode

        print("✅ Memory offloading patches applied for all components:")
        print("   • Text Encoder (T5) → CPU after prompt encoding")
        print("   • VAE Encoder → CPU after initial image encoding")
        print("   • CLIP → CPU after reference encoding")
        print("   • Transformer → CPU after sampling")
        print("   • VAE Decoder → CPU after video decode")

    def _init_preprocessing(self):
        """Initialize preprocessing components."""
        # Import preprocessing modules with proper path handling
        preprocess_dir = os.path.join(wan_dir, "wan", "modules", "animate", "preprocess")

        # Insert at the beginning to prioritize local imports
        if preprocess_dir not in sys.path:
            sys.path.insert(0, preprocess_dir)

        # Save original sys.path and working directory
        original_cwd = os.getcwd()
        original_sys_path = sys.path.copy()

        # Change to preprocess directory to ensure relative imports work
        os.chdir(preprocess_dir)

        # Ensure preprocessing directory is first in path (critical for production)
        sys.path = [preprocess_dir] + [p for p in sys.path if p != preprocess_dir]

        try:
            # Force load local utils module before process_pipepline imports it
            # This prevents it from finding /server/utils.py in production
            import importlib.util
            utils_spec = importlib.util.spec_from_file_location("utils", os.path.join(preprocess_dir, "utils.py"))
            utils_module = importlib.util.module_from_spec(utils_spec)
            sys.modules['utils'] = utils_module
            utils_spec.loader.exec_module(utils_module)

            # Temporarily disable SAM2 imports to avoid config errors
            # SAM2 is only needed for replacement mode anyway
            import types

            # Create mock SAM2 modules to prevent import errors
            mock_sam2 = types.ModuleType('sam2')
            mock_modeling = types.ModuleType('modeling')
            mock_sam = types.ModuleType('sam')
            mock_transformer = types.ModuleType('transformer')

            mock_transformer.USE_FLASH_ATTN = False
            mock_transformer.MATH_KERNEL_ON = True
            mock_transformer.OLD_GPU = True

            mock_sam.transformer = mock_transformer
            mock_modeling.sam = mock_sam
            mock_sam2.modeling = mock_modeling

            sys.modules['sam2'] = mock_sam2
            sys.modules['sam2.modeling'] = mock_modeling
            sys.modules['sam2.modeling.sam'] = mock_sam
            sys.modules['sam2.modeling.sam.transformer'] = mock_transformer

            # Also mock build_sam2_video_predictor
            import sam_utils_mock
            sys.modules['sam_utils'] = sam_utils_mock

            from process_pipepline import ProcessPipeline

            # Set up checkpoint paths
            pose2d_checkpoint = os.path.join(self.preprocess_ckpt_dir, 'pose2d/vitpose_h_wholebody.onnx')
            det_checkpoint = os.path.join(self.preprocess_ckpt_dir, 'det/yolov10m.onnx')
            sam2_checkpoint = os.path.join(self.preprocess_ckpt_dir, 'sam2/sam2_hiera_large.pt')
            flux_kontext_path = os.path.join(self.preprocess_ckpt_dir, 'FLUX.1-Kontext-dev')

            # Verify required checkpoints exist
            if not os.path.exists(pose2d_checkpoint):
                raise FileNotFoundError(f"Required checkpoint not found: {pose2d_checkpoint}")
            if not os.path.exists(det_checkpoint):
                raise FileNotFoundError(f"Required checkpoint not found: {det_checkpoint}")

            print(f"  ✓ Pose detection: {os.path.basename(pose2d_checkpoint)}")
            print(f"  ✓ Person detection: {os.path.basename(det_checkpoint)}")
            print(f"  {'✓' if os.path.exists(sam2_checkpoint) else '✗'} SAM2 (replacement mode): {'available' if os.path.exists(sam2_checkpoint) else 'not available - replacement mode disabled'}")
            print(f"  {'✓' if os.path.exists(flux_kontext_path) else '✗'} FLUX (enhanced retargeting): {'available' if os.path.exists(flux_kontext_path) else 'not available'}")

            # Initialize preprocessing pipeline
            # Note: SAM2 disabled for now due to config issues, only needed for replacement mode
            self.preprocess_pipeline = ProcessPipeline(
                det_checkpoint_path=det_checkpoint,
                pose2d_checkpoint_path=pose2d_checkpoint,
                sam_checkpoint_path=None,  # Disable SAM2 for now
                flux_kontext_path=flux_kontext_path if os.path.exists(flux_kontext_path) else None
            )

            self.sam2_available = os.path.exists(sam2_checkpoint)
        finally:
            # Restore original working directory and sys.path
            os.chdir(original_cwd)
            sys.path = original_sys_path

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run preprocessing + WAN animation generation on input data."""
        import shutil

        # Create temporary directory for all processing
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocess_dir = os.path.join(temp_dir, "preprocessed")
            os.makedirs(preprocess_dir, exist_ok=True)

            # Step 1: Run preprocessing
            print("🎬 Starting preprocessing...")
            replace_flag = input_data.mode == "replacement"

            # Check if replacement mode is requested but SAM2 is not available
            if replace_flag and not hasattr(self, 'sam2_available'):
                print("⚠️  WARNING: Replacement mode requested but SAM2 is not available.")
                print("    Falling back to animation mode.")
                print("    To use replacement mode, ensure SAM2 checkpoint is downloaded.")
                replace_flag = False

            self.preprocess_pipeline(
                video_path=input_data.driving_video.path,
                refer_image_path=input_data.reference_image.path,
                output_path=preprocess_dir,
                resolution_area=[input_data.resolution_width, input_data.resolution_height],
                fps=input_data.fps,
                iterations=input_data.mask_iterations,
                k=input_data.mask_kernel_size,
                w_len=input_data.mask_w_subdivisions,
                h_len=input_data.mask_h_subdivisions,
                retarget_flag=input_data.retarget_flag,
                use_flux=input_data.use_flux,
                replace_flag=replace_flag
            )
            print("✅ Preprocessing completed")

            # Verify preprocessed files exist
            required_files = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
            if replace_flag:
                required_files.extend(["src_bg.mp4", "src_mask.mp4"])

            for filename in required_files:
                filepath = os.path.join(preprocess_dir, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Preprocessing failed to generate: {filename}")

            # Step 2: Set random seed
            seed = input_data.seed
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            print(f"🎲 Using seed: {seed}")

            # Step 3: Generate animation
            print("🎨 Generating animation...")
            output_tensor = self.wan_animate.generate(
                src_root_path=preprocess_dir,
                replace_flag=replace_flag,
                clip_len=input_data.clip_len,
                refert_num=input_data.refert_num,
                shift=input_data.shift,
                sample_solver=input_data.sample_solver,
                sampling_steps=input_data.sampling_steps,
                guide_scale=input_data.guide_scale,
                input_prompt=input_data.input_prompt,
                n_prompt=input_data.n_prompt,
                seed=seed,
                offload_model=input_data.offload_model,
            )
            print("✅ Animation generation completed")

            # Step 4: Convert output tensor to video file with audio
            # Note: Memory offloading handled automatically by patches in _patch_generate_for_memory_efficiency()
            print("📹 Encoding output video...")
            temp_video_no_audio = os.path.join(temp_dir, "output_no_audio.mp4")

            # output_tensor is shape (C, N, H, W) - convert to (N, H, W, C)
            video_array = output_tensor.permute(1, 2, 3, 0).cpu().numpy()

            # Denormalize from [-1, 1] to [0, 255]
            video_array = ((video_array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            # Get video properties
            num_frames, height, width, channels = video_array.shape
            output_fps = input_data.fps if input_data.fps > 0 else 30

            # Write video using cv2 (temporary, without audio)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_no_audio, fourcc, output_fps, (width, height))

            for frame_idx in range(num_frames):
                frame = video_array[frame_idx]
                # Convert RGB to BGR for cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"✅ Video frames written: {num_frames} frames at {output_fps} FPS")

            # Step 5: Re-encode with browser-friendly codec (H.264) and add audio from input
            print("🎵 Adding audio and re-encoding with H.264...")
            final_output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

            import subprocess

            # Try to add audio from original driving video
            try:
                # ffmpeg command to:
                # 1. Take video from temp file
                # 2. Take audio from original driving video
                # 3. Encode with H.264 (libx264) - browser friendly
                # 4. Use AAC for audio - browser friendly
                # 5. Use yuv420p pixel format - maximum compatibility
                cmd = [
                    'ffmpeg',
                    '-i', temp_video_no_audio,  # Input video (no audio)
                    '-i', input_data.driving_video.path,  # Input audio source
                    '-map', '0:v:0',  # Take video from first input
                    '-map', '1:a:0?',  # Take audio from second input (if exists)
                    '-c:v', 'libx264',  # H.264 video codec
                    '-preset', 'medium',  # Encoding speed/quality balance
                    '-crf', '23',  # Quality (18-28, lower = better)
                    '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                    '-c:a', 'aac',  # AAC audio codec
                    '-b:a', '128k',  # Audio bitrate
                    '-shortest',  # Match shortest stream duration
                    '-y',  # Overwrite output
                    final_output_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    print("✅ Video encoded with H.264 + audio added")
                else:
                    # If ffmpeg fails (e.g., no audio in source), encode without audio
                    print(f"⚠️ Could not add audio, encoding without: {result.stderr[:200]}")
                    cmd_no_audio = [
                        'ffmpeg',
                        '-i', temp_video_no_audio,
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-pix_fmt', 'yuv420p',
                        '-y',
                        final_output_path
                    ]
                    subprocess.run(cmd_no_audio, check=True, timeout=300)
                    print("✅ Video encoded with H.264 (no audio)")

            except Exception as e:
                # Fallback: just copy the original file if ffmpeg fails
                print(f"⚠️ FFmpeg encoding failed: {e}, using original encoding")
                shutil.copy(temp_video_no_audio, final_output_path)

        return AppOutput(output_video=File(path=final_output_path))

    async def unload(self):
        """Clean up resources."""
        if hasattr(self, 'wan_animate'):
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del self.wan_animate