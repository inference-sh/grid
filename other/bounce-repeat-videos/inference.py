from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import os
import tempfile
from typing import Optional
import subprocess
import shutil
import math

class AppInput(BaseAppInput):
    video: File
    duration: Optional[float] = None  # Target duration in seconds (optional)

class AppOutput(BaseAppOutput):
    result: File

def run_ffmpeg_command(command):
    """Runs an ffmpeg command using subprocess and raises an error if it fails."""
    print(f"Running ffmpeg command: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    return process

def get_video_duration(video_path):
    """Gets the duration of a video file using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

class App(BaseApp):
    async def setup(self, metadata):
        """Check if ffmpeg is installed."""
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe are required but not found in the system PATH.")

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Create a bouncing video effect using ffmpeg.
        
        Args:
            input: Contains video file and optional target duration.
            
        Returns:
            AppOutput with the processed bouncing video.
        """
        # Get input parameters from the pydantic model
        input_path = input_data.video.path
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Define path for output
            bounce_output_path = os.path.join(tmpdir, "bounce_output.mp4")
            final_output_path = os.path.join(tmpdir, "final_output.mp4")

            try:
                # Get original duration
                original_duration = get_video_duration(input_path)
                
                # Use user-specified duration or default to 3x original
                target_duration = input_data.duration or (original_duration * 3)
                
                # Create the bounce effect in a single ffmpeg command using filter_complex
                # This creates a sequence of: original -> reversed -> original
                bounce_command = [
                    "ffmpeg",
                    "-i", input_path,
                    "-filter_complex", 
                    "[0:v]split[v1][v2];[v2]reverse[vr];[v1][vr]concat=n=2:v=1[vout]",
                    "-map", "[vout]",
                    bounce_output_path
                ]
                run_ffmpeg_command(bounce_command)
                
                # Get the duration of one bounce cycle
                bounce_duration = get_video_duration(bounce_output_path)
                
                # Calculate how many full bounce cycles we need
                num_cycles = math.ceil(target_duration / bounce_duration)
                
                # Create the final video with the exact number of cycles needed
                # We'll use stream_loop to repeat the bounce cycle
                loop_command = [
                    "ffmpeg",
                    "-stream_loop", str(num_cycles - 1),  # -1 because we count the first play
                    "-i", bounce_output_path,
                    "-t", str(target_duration),  # Ensure exact duration
                    "-c", "copy",
                    final_output_path
                ]
                run_ffmpeg_command(loop_command)

                # Create a persistent output file
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as final_output_file:
                    final_output_persistent_path = final_output_file.name
                shutil.copyfile(final_output_path, final_output_persistent_path)

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ffmpeg processing failed: {e.stderr}")
            except Exception as e:
                raise RuntimeError(f"An error occurred: {str(e)}")

        # Return the result
        return AppOutput(result=File(path=final_output_persistent_path))

