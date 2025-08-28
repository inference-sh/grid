import os
import subprocess
import tempfile
from pathlib import Path
import shutil
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import diffusers


class AppInput(BaseAppInput):
    video_path: File
    audio_path: File
    inference_steps: int = 20
    guidance_scale: float = 1.5

class AppOutput(BaseAppOutput):
    result_video: File

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the LatentSync model and resources."""

        print("Diffusers version:", diffusers.__version__)

        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Download required model files
        self._download_models()
                
        # Create soft links for auxiliary models
        self._create_soft_links()
        
        print("LatentSync model setup complete")

    def _download_models(self):
        """Download model checkpoints from HuggingFace using the huggingface_hub library."""
        try:
            from huggingface_hub import hf_hub_download
            import os
            
            # Create checkpoints directory structure
            os.makedirs("checkpoints/whisper", exist_ok=True)
            os.makedirs("checkpoints/auxiliary", exist_ok=True)
            
            # Download main model checkpoint
            print("Downloading LatentSync UNet model...")
            hf_hub_download(
                repo_id="ByteDance/LatentSync",
                filename="latentsync_unet.pt",
                local_dir="checkpoints",
                local_dir_use_symlinks=False
            )
            
            # Download Whisper model
            print("Downloading Whisper model...")
            hf_hub_download(
                repo_id="ByteDance/LatentSync",
                filename="whisper/tiny.pt",
                local_dir="checkpoints",
                local_dir_use_symlinks=False
            )
            
            # Download auxiliary models if needed
            auxiliary_files = [
                "auxiliary/2DFAN4-cd938726ad.zip",
                "auxiliary/s3fd-619a316812.pth",
                "auxiliary/vgg16-397923af.pth"
            ]
            
            for aux_file in auxiliary_files:
                try:
                    print(f"Downloading auxiliary model: {aux_file}")
                    hf_hub_download(
                        repo_id="ByteDance/LatentSync",
                        filename=aux_file,
                        local_dir="checkpoints",
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    print(f"Warning: Could not download auxiliary file {aux_file}: {e}")
                    # Continue with other files even if one fails
            
            print("Model checkpoints downloaded successfully")
        except Exception as e:
            print(f"Error downloading model checkpoints: {e}")
            raise

    def _create_soft_links(self):
        """Create soft links for auxiliary models."""
        os.makedirs(os.path.expanduser("~/.cache/torch/hub/checkpoints"), exist_ok=True)
        
        # We only need to create these links if the auxiliary models were downloaded
        aux_dir = Path("checkpoints/auxiliary")
        if aux_dir.exists():
            links = [
                ("2DFAN4-cd938726ad.zip", "2DFAN4-cd938726ad.zip"),
                ("s3fd-619a316812.pth", "s3fd-619a316812.pth"),
                ("vgg16-397923af.pth", "vgg16-397923af.pth")
            ]
            
            for src_name, dst_name in links:
                src_path = aux_dir / src_name
                dst_path = Path(os.path.expanduser(f"~/.cache/torch/hub/checkpoints/{dst_name}"))
                
                if src_path.exists() and not dst_path.exists():
                    try:
                        os.symlink(src_path, dst_path)
                    except Exception as e:
                        print(f"Warning: Could not create symlink {dst_path}: {e}")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run lip sync inference on the input video and audio."""
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save input files to temp directory
            video_path = os.path.join(temp_dir, "input_video.mp4")
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            output_path = os.path.join(temp_dir, "output_video.mp4")
            
            # Copy input files to temp paths
            shutil.copy(input_data.video_path.path, video_path)
            shutil.copy(input_data.audio_path.path, audio_path)
            
            # Get the path to the do_inference.py script
            # Assuming do_inference.py is in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            inference_script = os.path.join(current_dir, "do_inference.py")
            
            # Run inference using Python's exec function
            try:
                # Prepare arguments as they would be passed to the script
                import sys
                original_argv = sys.argv
                sys.argv = [
                    inference_script,
                    "--unet_config_path", os.path.join(current_dir, "configs", "unet", "second_stage.yaml"),
                    "--inference_ckpt_path", os.path.join(current_dir, "checkpoints", "latentsync_unet.pt"),
                    "--inference_steps", str(input_data.inference_steps),
                    "--guidance_scale", str(input_data.guidance_scale),
                    "--video_path", video_path,
                    "--audio_path", audio_path,
                    "--video_out_path", output_path,
                    "--seed", "1247"  # Default seed
                ]
                
                print(f"Running inference with args: {sys.argv}")
                
                # Execute the do_inference.py script
                with open(inference_script, 'r') as f:
                    script_content = f.read()
                
                # Add the current directory to the Python path if needed
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                    
                # Add LatentSync/latentsync to the Python path
                latentsync_dir = os.path.join(current_dir, "LatentSync")
                latentsync_module_dir = os.path.join(latentsync_dir, "latentsync")
                if os.path.exists(latentsync_module_dir) and latentsync_module_dir not in sys.path:
                    sys.path.append(latentsync_dir)
                    print(f"Added {latentsync_dir} to Python path")
                    
                # Execute the script
                exec(script_content, {'__name__': '__main__'})
                
                # Restore original argv
                sys.argv = original_argv
                
                if not os.path.exists(output_path):
                    raise RuntimeError(f"Output video was not generated.")
                
                # Create a permanent copy of the output
                result_path = "/tmp/lipsync_result.mp4"
                shutil.copy(output_path, result_path)
                
                return AppOutput(result_video=File(path=result_path))
                
            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()
                raise

    async def unload(self):
        """Clean up resources."""
        # No specific cleanup needed
        pass