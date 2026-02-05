import os
import subprocess
from typing import Annotated
from pydantic import Field
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

class AppInput(BaseAppInput):
    video_file: Annotated[File, Field(
        description="The video file to process"
    )]

class AppOutput(BaseAppOutput):
    audio_file: Annotated[File, Field(
        description="The extracted audio from the video"
    )]
    silent_video: Annotated[File, Field(
        description="The video file with audio removed"
    )]

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Extract audio from video and create silent video."""
        # Define output paths
        video_path = input_data.video_file.path
        filename = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = f"/tmp/{filename}_audio.mp3"
        silent_video_path = f"/tmp/{filename}_silent.mp4"
        
        # Extract audio from video
        subprocess.run([
            "ffmpeg", "-i", video_path, 
            "-q:a", "0", "-map", "a", 
            audio_path
        ], check=True)
        
        # Create video without audio
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-c:v", "copy", "-an",
            silent_video_path
        ], check=True)
        
        return AppOutput(
            audio_file=File(path=audio_path),
            silent_video=File(path=silent_video_path)
        )

    async def unload(self):
        """Clean up resources here."""
        pass