from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from typing import List
import cv2
import numpy as np

class AppInput(BaseAppInput):
    video_paths: List[File]

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    async def setup(self):
        """Initialize your model and resources here."""
        print("Setting up video stitching app")
        print("Setup complete")
        

    async def run(self, input_data: AppInput) -> AppOutput:
        """Run prediction on the input data."""
        # Open all video captures first
        caps = [cv2.VideoCapture(str(path.path)) for path in input_data.video_paths]
        
        # Get dimensions of first video
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(caps[0].get(cv2.CAP_PROP_FPS))
        
        # Calculate grid dimensions
        n_videos = len(caps)
        grid_size = int(np.ceil(np.sqrt(n_videos)))
        
        # Create video writer with combined dimensions
        combined_width = width * grid_size
        combined_height = height * grid_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('result.mp4', fourcc, fps, (combined_width, combined_height))
        
        while True:
            # Read frames from all videos
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if len(frames) != len(caps):
                break
                
            # Create empty combined frame
            combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Place frames in grid
            for idx, frame in enumerate(frames):
                i = (idx // grid_size) * height
                j = (idx % grid_size) * width
                combined_frame[i:i+height, j:j+width] = frame
            
            # Write combined frame
            writer.write(combined_frame)
        
        # Clean up
        for cap in caps:
            cap.release()
        writer.release()
        
        return AppOutput(result=File(path="result.mp4"))

    async def unload(self):
        """Clean up resources here."""
        pass