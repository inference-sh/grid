from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from typing import List, Tuple, Dict, Optional, Union
from pydantic import Field
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import tempfile
import os
import requests
from enum import Enum
from .fonts import FONTS

class GoogleFont(str, Enum):
    ROBOTO = "Roboto"
    OPEN_SANS = "Open Sans"
    LATO = "Lato"
    MONTSERRAT = "Montserrat"
    OSWALD = "Oswald"
    SOURCE_SANS_PRO = "Source Sans Pro"
    RALEWAY = "Raleway"
    UBUNTU = "Ubuntu"
    MERRIWEATHER = "Merriweather"
    PLAYFAIR_DISPLAY = "Playfair Display"
    POPPINS = "Poppins"
    NOTO_SANS = "Noto Sans"
    RUBIK = "Rubik"
    ALEF = "Alef"
    WORK_SANS = "Work Sans"
    NUNITO = "Nunito"
    FIRA_SANS = "Fira Sans"
    QUICKSAND = "Quicksand"
    PT_SANS = "PT Sans"

class TextPosition(str, Enum):
    CENTER_BOTTOM = "center-bottom"
    CENTER_TOP = "center-top"
    CENTER_CENTER = "center-center"
    LEFT_BOTTOM = "left-bottom"
    LEFT_TOP = "left-top"
    LEFT_CENTER = "left-center"
    RIGHT_BOTTOM = "right-bottom"
    RIGHT_TOP = "right-top"
    RIGHT_CENTER = "right-center"

class TextAlign(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

# Function to get font URL from FONTS data
def get_font_url(font_name):
    """Get the URL for a font by name from the FONTS data"""
    for item in FONTS["items"]:
        if item["family"].lower() == font_name.lower():
            # Get the regular variant if available, otherwise get the first available variant
            if "regular" in item["files"]:
                return item["files"]["regular"]
            else:
                # Get the first available font variant
                for variant in item["variants"]:
                    if variant in item["files"]:
                        return item["files"][variant]
    return None

# Build the FONT_URL_MAP dictionary dynamically
FONT_URL_MAP = {}
# Fallback URLs in case we can't find a font in the FONTS data
FALLBACK_URLS = {
    GoogleFont.ROBOTO: "http://fonts.gstatic.com/s/roboto/v20/KFOmCnqEu92Fr1Mu4mxP.ttf",
    GoogleFont.OPEN_SANS: "http://fonts.gstatic.com/s/opensans/v18/mem8YaGs126MiZpBA-UFVZ0e.ttf",
    GoogleFont.LATO: "http://fonts.gstatic.com/s/lato/v17/S6uyw4BMUTPHjx4wWw.ttf"
}

for font in GoogleFont:
    url = get_font_url(font.value)
    if url:
        FONT_URL_MAP[font] = url
    elif font in FALLBACK_URLS:
        FONT_URL_MAP[font] = FALLBACK_URLS[font]
    else:
        # If we can't find the font, use Roboto as a fallback
        fallback_url = FALLBACK_URLS[GoogleFont.ROBOTO]
        FONT_URL_MAP[font] = fallback_url

class AppInput(BaseAppInput):
    segments: List = Field(
        description="Array of segments with start/end timestamps and text for video captions. Format: [{'start': timestamp_start, 'end': timestamp_end, 'text': text_to_show}]"
    )
    video_file: File = Field(
        description="Video file to add captions to"
    )
    font: GoogleFont = Field(
        default=GoogleFont.ROBOTO,
        description="Select a font for the captions"
    )
    font_size: int = Field(
        default=28, 
        description="Font size for captions"
    )
    font_color: str = Field(
        default="white", 
        description="Font color (name or hex code)"
    )
    bg_color: Optional[str] = Field(
        default='transparent', 
        description="Background color behind text"
    )
    position: TextPosition = Field(
        default=TextPosition.CENTER_BOTTOM, 
        description="Position of captions on screen"
    )
    stroke_color: Optional[str] = Field(
        default="black", 
        description="Text outline color for better visibility"
    )
    stroke_width: int = Field(
        default=1, 
        description="Width of text outline in pixels"
    )
    margin_horizontal: int = Field(
        default=0,
        description="Horizontal margin from the edges in pixels"
    )
    margin_vertical: int = Field(
        default=20, 
        description="Vertical margin from the edges in pixels"
    )
    text_align: TextAlign = Field(
        default=TextAlign.CENTER, 
        description="Text alignment within caption box"
    )
    fix_whisper_30s_timestamps: bool = Field(
        default=False, 
        description="Whisper elaborates 30s chunks of audio. This option will fix the timestamps to ensure the captions are in sync with the audio."
    )


class AppOutput(BaseAppOutput):
    captioned_video: File = Field(
        description="Video with captions added"
    )

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize resources."""
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Add captions to the video based on the timestamps in the segments array."""
        # Use the segments directly from input
        segments_data = input_data.segments
        
        # Load the video
        video = VideoFileClip(input_data.video_file.path)
                
        # Create margin tuple (horizontal, vertical)
        margin_tuple = (input_data.margin_horizontal, input_data.margin_vertical)
        
        # Determine font path
        font_path = 'Arial.otf'  # Default font
        
        # Download the selected Google font
        try:
            # Get the URL from the mapping dictionary using the enum value
            font_url = FONT_URL_MAP[input_data.font]
            response = requests.get(font_url)
            response.raise_for_status()  # Ensure the request was successful
            
            # Create a temporary file to save the font
            temp_font_file = tempfile.NamedTemporaryFile(suffix=".ttf", delete=False)
            temp_font_file.write(response.content)
            temp_font_file.close()
            
            font_path = temp_font_file.name
        except Exception as e:
            print(f"Error downloading font, using default: {e}")
        
        # Create TextClips for each caption and add them to the video
        text_clips = []
        
        prev_end_time = 0

        extra_seconds = 30

        for segment in segments_data:
            text = segment["text"]
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Handle incomplete timestamps
            if start_time is None:
                start_time = 0
                print(f"Using video start for caption start_time: {text}")
                
            # If end_time is None, use the video duration
            if end_time is None:
                end_time = video.duration
                print(f"Using video duration for caption end_time: {text}")
                
            if end_time < prev_end_time and input_data.fix_whisper_30s_timestamps:
                prev_end_time = end_time
                end_time += extra_seconds
                start_time += extra_seconds
                extra_seconds += 30
            
            # Create a TextClip with positioning parameters
            txt_clip = TextClip(
                text=text,
                font=font_path,  # Use downloaded font or default
                font_size=input_data.font_size,
                color=input_data.font_color,
                bg_color=input_data.bg_color if input_data.bg_color != 'transparent' else None,
                stroke_color=input_data.stroke_color,
                stroke_width=input_data.stroke_width,
                interline=input_data.font_size * 0.2,
                method='caption',  # Wrap text to fit video width
                size=(video.w - 2*input_data.margin_horizontal, None),  # Width with margin, auto height
                text_align=input_data.text_align.value,  # Alignment within the text block
                margin=margin_tuple,  # Margin around text as a tuple (horizontal, vertical)
            ).with_position(input_data.position.value.split('-'))

            # Set the duration of the text (from start to end timestamp)
            txt_clip = txt_clip.with_start(start_time).with_end(end_time)
            
            text_clips.append(txt_clip)
        
        # Overlay the text clips on the video
        final_video = CompositeVideoClip([video] + text_clips)
        
        # Create a temporary file for the output
        output_path = tempfile.mktemp(suffix=".mp4")
        
        # Write the final video to file
        final_video.write_videofile(output_path, codec="libx264")
        
        # Close the clips to free resources
        video.close()
        final_video.close()
        
        # Clean up the temporary font file if we created one
        if os.path.exists(font_path) and font_path != 'Arial.otf':
            os.unlink(font_path)
        
        return AppOutput(captioned_video=File(path=output_path))

