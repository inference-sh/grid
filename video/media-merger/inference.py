from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import os
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips

# Import all available effects that we'll use
from moviepy.video.fx.Resize import Resize
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from moviepy.video.fx.CrossFadeIn import CrossFadeIn
from moviepy.video.fx.CrossFadeOut import CrossFadeOut
from moviepy.video.fx.Rotate import Rotate
from moviepy.video.fx.SlideIn import SlideIn
from moviepy.video.fx.BlackAndWhite import BlackAndWhite
from moviepy.video.fx.MirrorX import MirrorX
from moviepy.video.fx.MirrorY import MirrorY
from moviepy.video.fx.InvertColors import InvertColors
from moviepy.video.fx.AccelDecel import AccelDecel

from enum import Enum
import numpy as np

transition_types = [
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
        default="crossfade",
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
    output_format: str = Field(
        default="mp4",
        description="Format of the output video file",
        examples=["mp4", "avi", "mov", "webm"]
    )
    fps: Optional[int] = Field(
        default=30,
        description="Frames per second for the output video",
    )

class AppOutput(BaseAppOutput):
    result: File = Field(
        description="The resulting merged video file with applied transitions"
    )

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your model and resources here."""
        # Create a temp directory for processing
        self.temp_dir = tempfile.mkdtemp()
        
    def apply_transition(self, clip1, clip2, transition_type, duration=1.0):
        """Apply the specified transition between two clips."""
        w, h = clip1.size  # Get dimensions from the first clip

        if clip2 is None:
            return clip1.with_effects([FadeOut(duration)])
        
        # For transitions that need clips to overlap
        if transition_type in [
            "crossfade", 
            "fade_to_black",
        ]:
            # Create the transitioned clips
            if transition_type == "crossfade":
                clip1_out = clip1.with_effects([CrossFadeOut(duration)])
                clip2_in = clip2.with_effects([CrossFadeIn(duration)])
                
                # Position clip2 to start before clip1 ends to create overlap
                clip2_start = clip1.duration - duration
                
                # Create the composite clip with the overlapping transition
                composite = CompositeVideoClip([
                    clip1_out,
                    clip2_in.with_start(clip2_start)
                ])

            
            elif transition_type == "fade_to_black":
                clip1_out = clip1.with_effects([FadeOut(duration)])
                clip2_in = clip2.with_effects([FadeIn(duration)])
                
                # Create the composite clip with the overlapping transition
                composite = concatenate_videoclips([
                    clip1_out,
                    clip2_in
                ])


            
            # Ensure the composite maintains the correct size
            if composite.size != (w, h):
                composite = composite.with_effects([Resize(width=w, height=h)])
                
            return composite
        
        # For slide transitions
        elif transition_type in [
            "slide_left",
            "slide_right",
            "slide_up",
            "slide_down"
        ]:
            # Ensure clip2 has the same dimensions as clip1
            if clip2.size != (w, h):
                clip2 = clip2.with_effects([Resize(width=w, height=h)])
                
            # Determine slide direction
            if transition_type == "slide_left":
                side = 'right'
            elif transition_type == "slide_right":
                side = 'left'
            elif transition_type == "slide_up":
                side = 'bottom'
            elif transition_type == "slide_down":
                side = 'top'
            
            # Apply slide effect to the second clip
            clip2_with_slide = clip2.with_effects([SlideIn(duration, side=side)])
            
            # Create and return a composite with the slide effect
            clip2_start = clip1.duration - duration
            composite = CompositeVideoClip([
                clip1,
                clip2_with_slide.with_start(clip2_start)
            ])
            
            # Ensure the composite maintains the correct size
            if composite.size != (w, h):
                composite = composite.with_effects([Resize(width=w, height=h)])
                
            return composite
        
        # Default fallback - crossfade
        composite = CompositeVideoClip([
            clip1.with_effects([CrossFadeOut(duration)]),
            clip2.with_effects([CrossFadeIn(duration)]).with_start(clip1.duration - duration)
        ])
        
        # Ensure the composite maintains the correct size
        if composite.size != (w, h):
            composite = composite.with_effects([Resize(width=w, height=h)])
            
        return composite

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run prediction on the input data."""
        clips = []
        base_width, base_height = None, None
        
        # Process each media file
        for i, media in enumerate(input_data.media_files):
            file_path = media.file.path
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Handle different file types
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                clip = VideoFileClip(file_path)
                # Set duration if specified and not -1
                if media.duration is not None and media.duration > 0:
                    clip = clip.subclipped(0, media.duration)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                # For images, create a clip with the specified duration (default 5 seconds if -1)
                image_duration = 5.0 if media.duration is None or media.duration < 0 else media.duration
                clip = ImageClip(file_path, duration=image_duration)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Store the first clip's dimensions to use for all clips
            if i == 0:
                base_width, base_height = clip.size
            else:
                # Resize all subsequent clips to match the first clip's dimensions
                clip = clip.with_effects([Resize(width=base_width, height=base_height)])
            
            clips.append(clip)
        
        # Prepare final video with transitions
        result_clip = None
        
        # Process clips incrementally, applying transitions one by one
        for i in range(len(clips)):
            current_clip = clips[i]
            
            if result_clip is None:
                # First clip becomes our initial result
                result_clip = current_clip
            else:
                # Apply transition between current result and next clip
                transition_type = input_data.media_files[i-1].transition_type
                transition_duration = input_data.media_files[i-1].transition_duration or 1.0
                
                # Calculate safe transition duration
                transition_duration = min(
                    transition_duration, 
                    result_clip.duration/2, 
                    current_clip.duration/2
                )
                
                # Apply transition and update result
                result_clip = self.apply_transition(
                    result_clip, current_clip, transition_type, transition_duration
                )
                
                # Ensure the composite clip maintains the base dimensions
                if result_clip.size != (base_width, base_height):
                    result_clip = result_clip.with_effects([Resize(width=base_width, height=base_height)])
        
        final_video = result_clip

        # Save the output
        output_path = os.path.join(self.temp_dir, f"result.{input_data.output_format}")
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac", 
            ffmpeg_params=[
                "-profile:v", "main",  # Critical for Safari
                "-pix_fmt", "yuv420p",  # Critical for Safari
                "-movflags", "+faststart",  # Helps with streaming
                "-crf", "23"  # Reasonable quality
            ],
            fps=input_data.fps or 30,
            logger=None  # MoviePy 2.x changed logger behavior
        )
        
        
        # Close all clips to release resources
        for clip in clips:
            if hasattr(clip, 'close'):
                clip.close()
        
        if hasattr(final_video, 'close'):
            final_video.close()
        
        return AppOutput(result=File(path=output_path))

    async def unload(self):
        """Clean up resources here."""
        import shutil
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)