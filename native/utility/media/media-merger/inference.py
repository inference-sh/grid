from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import os
import re
import subprocess
import tempfile

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
DEFAULT_IMAGE_DURATION = 5.0
# xfade names for the overlapping transitions. The cover* modes keep the
# outgoing clip still while the incoming one slides over it.
XFADE = {
    "crossfade": "fade",
    "slide_left": "coverleft",
    "slide_right": "coverright",
    "slide_up": "coverup",
    "slide_down": "coverdown",
}

transition_types = [
    "no-transition",
    "crossfade",
    "slide_left",
    "slide_right",
    "slide_up",
    "slide_down",
    "fade_to_black"
]
    
class Media(BaseModel):
    file: File = Field(
        description="The media file (video or image) to include in the sequence"
    )
    transition_type: str = Field(
        default=transition_types[0],
        enum=transition_types,
        description="Type of transition to apply between this media and the next one (if there is no next clip it will not be applied)"
    )
    duration: Optional[float] = Field(
        default=-1.0,
        description="Duration in seconds for this media. Use -1 to use entire duration for videos or 5 seconds for images.",
    )
    transition_duration: Optional[float] = Field(
        default=1.0,
        description="Duration in seconds for the transition effect",
    )

class AppInput(BaseAppInput):
    media_files: List[Media] = Field(
        description="List of media files to merge with transitions",
    )
    output_format: Literal["mp4", "mov", "avi"] = Field(
        default="mp4",
        description="Format of the output video file",
    )
    fps: Optional[int] = Field(
        default=30,
        description="Frames per second for the output video",
    )

class AppOutput(BaseAppOutput):
    result: File = Field(
        description="The resulting merged video file with applied transitions"
    )

def probe(path):
    """Return (width, height, duration, has_audio) parsed from `ffmpeg -i`."""
    info = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True
    ).stderr
    size = re.search(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5})", info)
    if size is None:
        raise ValueError(f"No video stream in {os.path.basename(path)}")
    width, height = int(size[1]), int(size[2])
    rotation = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", info)
    if rotation and round(float(rotation[1])) % 180 == 90:
        width, height = height, width
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", info)
    seconds = None
    if duration:
        seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    return width, height, seconds, re.search(r"Stream #.*Audio:", info) is not None


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        # Create a temp directory for processing
        self.temp_dir = tempfile.mkdtemp()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        if not input_data.media_files:
            raise ValueError("media_files is empty")
        fps = input_data.fps or 30
        inputs, graph, durations, audio = [], [], [], []
        width, height = None, None

        for i, media in enumerate(input_data.media_files):
            file_path = media.file.path
            file_ext = os.path.splitext(file_path)[1].lower()
            is_image = file_ext in IMAGE_EXTS
            if not is_image and file_ext not in VIDEO_EXTS:
                raise ValueError(f"Unsupported file format: {file_ext}")

            clip_width, clip_height, duration, has_audio = probe(file_path)
            if media.duration is not None and media.duration > 0:
                duration = media.duration
            elif is_image:
                duration = DEFAULT_IMAGE_DURATION
            if not duration:
                raise ValueError(f"Could not read the duration of {os.path.basename(file_path)}")

            # The first clip sets the output size; libx264 yuv420p needs even dimensions
            if i == 0:
                width, height = clip_width - clip_width % 2, clip_height - clip_height % 2

            inputs += ["-i", file_path]
            # Images hold their first frame, with any transparency over black
            still = "trim=end_frame=1,format=rgba,premultiply=inplace=1," if is_image else ""
            graph.append(
                f"[{i}:v]{still}scale={width}:{height},setsar=1,format=yuv420p,"
                f"tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},"
                f"setpts=PTS-STARTPTS,fps={fps}[v{i}]"
            )
            if has_audio:
                audio.append((i, duration))
            durations.append(duration)

        # Join the clips one by one. Each transition belongs to the clip before it.
        video, total, starts = "v0", durations[0], [0.0]
        for i in range(1, len(durations)):
            previous = input_data.media_files[i - 1]
            kind = previous.transition_type
            overlap = min(previous.transition_duration or 1.0, total / 2, durations[i] / 2)
            if kind == "no-transition":
                graph.append(f"[{video}][v{i}]concat=n=2:v=1:a=0[m{i}]")
                start = total
            elif kind == "fade_to_black":
                graph.append(f"[{video}]fade=t=out:st={total - overlap}:d={overlap}[o{i}]")
                graph.append(f"[v{i}]fade=t=in:st=0:d={overlap}[i{i}]")
                graph.append(f"[o{i}][i{i}]concat=n=2:v=1:a=0[m{i}]")
                start = total
            else:
                start = total - overlap
                graph.append(
                    f"[{video}][v{i}]xfade=transition={XFADE.get(kind, 'fade')}:"
                    f"duration={overlap}:offset={start}[m{i}]"
                )
            video, total = f"m{i}", start + durations[i]
            starts.append(start)

        # Audio plays from each clip's start; overlapping clips mix unattenuated
        maps = ["-map", f"[{video}]"]
        if audio:
            for i, duration in audio:
                graph.append(
                    f"[{i}:a]atrim=duration={duration},asetpts=PTS-STARTPTS,"
                    f"aresample=44100,aformat=channel_layouts=stereo,"
                    f"adelay={round(starts[i] * 1000)}:all=1[a{i}]"
                )
            labels = "".join(f"[a{i}]" for i, _ in audio)
            graph.append(f"{labels}amix=inputs={len(audio)}:normalize=0:dropout_transition=0[audio]")
            maps += ["-map", "[audio]"]

        output_path = os.path.join(self.temp_dir, f"result.{input_data.output_format}")
        result = subprocess.run(
            [
                FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                *inputs,
                "-filter_complex", ";".join(graph),
                *maps,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-profile:v", "main",  # Critical for Safari
                "-pix_fmt", "yuv420p",  # Critical for Safari
                "-movflags", "+faststart",  # Helps with streaming
                "-crf", "23",  # Reasonable quality
                "-t", str(total),
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-2000:]}")

        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        import shutil
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
