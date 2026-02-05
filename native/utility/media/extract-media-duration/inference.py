from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
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
        
        # Use shell command to extract duration
        cmd = f"ffmpeg -i {input_data.media_file.path} 2>&1 | grep \"Duration\" | cut -d ' ' -f 4 | sed s/,// | sed 's@\\..*@@g' | awk '{{ split($1, A, \":\"); split(A[3], B, \".\"); print 3600*A[1] + 60*A[2] + B[1] }}'"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 or not result.stdout.strip():
            raise Exception(f"Error processing media file: {result.stderr}")
        
        # Parse the output to get duration in seconds
        duration_seconds = float(result.stdout.strip())
        
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