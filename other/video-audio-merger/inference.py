import os
import subprocess
from typing import Annotated
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

class AppInput(BaseAppInput):
    video_file: Annotated[File, Field(
        description="The video file to process"
    )]
    audio_file: Annotated[File, Field(
        description="The audio file to merge with the video"
    )]
    preserve_original_audio: Annotated[bool, Field(
        description="Whether to preserve the original video audio and add the new audio on top (True) or replace it entirely (False)",
        default=False
    )]

class AppOutput(BaseAppOutput):
    merged_video: Annotated[File, Field(
        description="The video with merged audio"
    )]

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Merge audio with video."""
        # Define output paths
        video_path = input_data.video_file.path
        audio_path = input_data.audio_file.path
        filename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"/tmp/{filename}_merged.mp4"
        
        if input_data.preserve_original_audio:
            # Merge audio with original audio preserved (mix them)
            # The -shortest flag makes the output as long as the shortest input
            subprocess.run([
                "ffmpeg", "-i", video_path, "-i", audio_path,
                "-filter_complex", "amix=inputs=2:duration=shortest",
                "-c:v", "copy", output_path
            ], check=True)
        else:
            # Replace original audio with new audio
            subprocess.run([
                "ffmpeg", "-i", video_path, "-i", audio_path,
                "-map", "0:v", "-map", "1:a", 
                "-c:v", "copy", "-shortest",
                output_path
            ], check=True)
        
        return AppOutput(
            merged_video=File(path=output_path)
        )

    async def unload(self):
        """Clean up resources here."""
        pass