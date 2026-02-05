from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import Optional
import cv2
import numpy as np
import os

class AppInput(BaseAppInput):
    video_file: File = Field(description="The input video file to extract frames from (MP4, AVI, MOV, etc.)")
    nth_last_frame: int = Field(default=1, description="Extract the nth frame from the end of the video (1 = last frame, 2 = second-to-last, etc.)")

class AppOutput(BaseAppOutput):
    extracted_frame: File = Field(description="The extracted nth last frame from the video as a PNG image file")
    total_frames: int = Field(description="Total number of frames in the input video")
    frame_index: int = Field(description="The actual frame index that was extracted (0-based)")
    video_info: str = Field(description="Basic information about the processed video (duration, fps, resolution)")

# For LLM apps, you can use the LLMInput and LLMInputWithImage classes for convenience
# from inferencesh import LLMInput, LLMInputWithImage
# The LLMInput class provides a standard structure for LLM-based applications with:
# - system_prompt: Sets the AI assistant's role and behavior
# - context: List of previous conversation messages between user and assistant
# - text: The current user's input prompt
#
# Example usage:
# class AppInput(LLMInput):
#     additional_field: str = Field(description="Any additional input needed")

# The LLMInputWithImage class extends LLMInput to support image inputs by adding:
# - image: Optional File field for providing images to vision-capable models
#
# Example usage:
# class AppInput(LLMInputWithImage):
#     additional_field: str = Field(description="Any additional input needed")

# Each ContextMessage in the context list contains:
# - role: Either "user", "assistant", or "system"
# - text: The message content
#
# ContextMessageWithImage adds:
# - image: Optional File field for messages containing images



class App(BaseApp):
    async def setup(self, metadata):
        """Initialize your resources here."""
        # No special setup needed for video frame extraction
        pass

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Extract the nth last frame from a video file."""
        # Check if input video file exists and is accessible
        if not input_data.video_file.exists():
            raise RuntimeError(f"Video file does not exist at path: {input_data.video_file.path}")
        
        # Validate nth_last_frame parameter
        if input_data.nth_last_frame < 1:
            raise ValueError("nth_last_frame must be a positive integer (1 = last frame, 2 = second-to-last, etc.)")
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(input_data.video_file.path)
            
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {input_data.video_file.path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames == 0:
                raise ValueError("Video file has no frames or could not determine frame count")
            
            # Check if nth_last_frame is valid
            if input_data.nth_last_frame > total_frames:
                raise ValueError(f"Requested nth_last_frame ({input_data.nth_last_frame}) exceeds total frames ({total_frames})")
            
            # Calculate the target frame index (0-based)
            target_frame_index = total_frames - input_data.nth_last_frame
            
            # Set video position to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret:
                raise RuntimeError(f"Could not read frame at index {target_frame_index}")
            
            # Create output filename
            base_filename = os.path.splitext(input_data.video_file.filename or "video")[0]
            output_filename = f"{base_filename}_frame_{target_frame_index}_nth_last_{input_data.nth_last_frame}.png"
            output_path = f"/tmp/{output_filename}"
            
            # Save the frame as PNG
            success = cv2.imwrite(output_path, frame)
            
            if not success:
                raise RuntimeError(f"Failed to save frame to {output_path}")
            
            # Create video info string
            video_info = f"Resolution: {width}x{height}, FPS: {fps:.2f}, Duration: {duration:.2f}s, Total frames: {total_frames}"
            
            # Release video capture
            cap.release()
            
            return AppOutput(
                extracted_frame=File(path=output_path),
                total_frames=total_frames,
                frame_index=target_frame_index,
                video_info=video_info
            )
            
        except cv2.error as e:
            raise RuntimeError(f"OpenCV error while processing video: {str(e)}")
        except Exception as e:
            # Clean up video capture if it was opened
            if 'cap' in locals():
                cap.release()
            raise RuntimeError(f"Error processing video: {str(e)}")

    async def unload(self):
        """Clean up resources here."""
        # No special cleanup needed
        pass