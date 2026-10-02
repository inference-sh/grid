from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
import re
import subprocess

class AppInput(BaseAppInput):
    media_file: File

class AppOutput(BaseAppOutput):
    duration: float
    duration_formatted: str

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Extract the duration of the input media file using ffmpeg."""
        
        # ffmpeg prints "Duration: HH:MM:SS.xx" to stderr; parse it without a shell.
        result = subprocess.run(
            ["ffmpeg", "-i", input_data.media_file.path], capture_output=True, text=True
        )
        output = result.stderr + result.stdout
        line = next((l for l in output.splitlines() if "Duration" in l), None)
        if line is None:
            raise Exception(f"Error processing media file: {result.stderr}")

        # Whole seconds, matching the previous `cut | sed | awk` pipeline (fraction dropped).
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)", line)
        duration_seconds = (
            float(int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3)))
            if match else 0.0
        )

        # Format duration as HH:MM:SS
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        return AppOutput(
            duration=duration_seconds,
            duration_formatted=duration_formatted
        )

    async def unload(self):
        """Clean up resources here."""
        pass