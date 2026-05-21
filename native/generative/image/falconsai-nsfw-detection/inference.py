import os
import logging
from typing import List

import torch
from PIL import Image
from pydantic import Field
from transformers import ViTImageProcessor, AutoModelForImageClassification

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameResult(BaseAppOutput):
    """Result for a single frame"""
    frame_index: int = Field(description="Frame index in the video")
    timestamp_seconds: float = Field(description="Timestamp in seconds")
    classification: str = Field(description="NSFW or SAFE")
    nsfw_score: float = Field(description="NSFW probability score (0-1)")
    safe_score: float = Field(description="Safe probability score (0-1)")


class AppInput(BaseAppInput):
    file: File = Field(description="Input image or video file to check for NSFW content")
    frame_interval_seconds: float = Field(
        default=1.0,
        description="For videos: sample one frame every N seconds (default: 1.0)"
    )
    max_frames: int = Field(
        default=2,
        description="Maximum number of frames to analyze (default: 1)"
    )
    nsfw_threshold: float = Field(
        default=0.5,
        description="Threshold for NSFW classification (0-1, default: 0.5)"
    )


class AppOutput(BaseAppOutput):
    is_nsfw: bool = Field(description="Whether NSFW content was detected")
    overall_classification: str = Field(description="Overall classification: NSFW or SAFE")
    max_nsfw_score: float = Field(description="Maximum NSFW score found across all frames")
    frames_analyzed: int = Field(description="Number of frames analyzed")
    nsfw_frames_count: int = Field(description="Number of frames classified as NSFW")
    frame_results: List[FrameResult] = Field(description="Detailed results for each analyzed frame")


class App(BaseApp):
    async def setup(self, metadata):
        """Initialize model and processor once"""
        logger.info("Loading Falconsai NSFW detection model...")
        
        # Use accelerate for device management
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        
        # Load processor and model
        self.processor = ViTImageProcessor.from_pretrained(
            "Falconsai/nsfw_image_detection"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "Falconsai/nsfw_image_detection"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")

    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a video based on extension"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        ext = os.path.splitext(file_path)[1].lower()
        return ext in video_extensions

    def _extract_frames_from_video(self, video_path: str, interval_seconds: float, max_frames: int) -> List[tuple]:
        """Extract frames from video at specified intervals, capped by max_frames.
        
        If max_frames would result in fewer samples than interval-based sampling,
        we take max_frames samples at equal distances throughout the video.
        
        Returns list of (frame_index, timestamp, pil_image) tuples.
        """
        import cv2
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video info: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
        
        # Calculate frame interval based on time
        frame_interval = int(fps * interval_seconds) if fps > 0 else 30
        frame_interval = max(1, frame_interval)
        
        # Calculate how many frames we'd get with interval-based sampling
        estimated_frames = (total_frames // frame_interval) + 1 if total_frames > 0 else 1
        
        # If max_frames is smaller, recalculate interval to get equal distance samples
        if max_frames > 0 and estimated_frames > max_frames:
            # Distribute max_frames equally across the video
            frame_interval = total_frames // max_frames if max_frames > 1 else total_frames
            frame_interval = max(1, frame_interval)
            logger.info(f"Capping to {max_frames} frames, adjusted interval to {frame_interval} frames")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Stop if we've reached max_frames
                if max_frames > 0 and len(frames) >= max_frames:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                timestamp = frame_idx / fps if fps > 0 else 0
                frames.append((frame_idx, timestamp, pil_image))
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        
        return frames

    def _classify_images(self, pil_images: List[Image.Image], threshold: float) -> List[dict]:
        """Classify a batch of images for NSFW content."""
        if not pil_images:
            return []
        
        # Process images
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        results = []
        for prob in probs:
            # Model outputs: index 0 = normal/safe, index 1 = nsfw
            safe_score = prob[0].item()
            nsfw_score = prob[1].item()
            
            classification = "NSFW" if nsfw_score > threshold else "SAFE"
            results.append({
                "classification": classification,
                "nsfw_score": nsfw_score,
                "safe_score": safe_score
            })
        
        return results

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run NSFW detection on input image or video."""
        
        if not input_data.file.exists():
            raise RuntimeError(f"Input file does not exist at path: {input_data.file.path}")
        
        file_path = input_data.file.path
        threshold = input_data.nsfw_threshold
        
        frame_results = []
        
        if self._is_video_file(file_path):
            # Process video
            logger.info(f"Processing video file: {file_path}")
            frames_data = self._extract_frames_from_video(
                file_path, 
                input_data.frame_interval_seconds,
                input_data.max_frames
            )
            
            # Process frames in batches to manage memory
            batch_size = 8
            for i in range(0, len(frames_data), batch_size):
                batch = frames_data[i:i + batch_size]
                pil_images = [f[2] for f in batch]
                
                classifications = self._classify_images(pil_images, threshold)
                
                for j, (frame_idx, timestamp, _) in enumerate(batch):
                    cls_result = classifications[j]
                    frame_results.append(FrameResult(
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp,
                        classification=cls_result["classification"],
                        nsfw_score=cls_result["nsfw_score"],
                        safe_score=cls_result["safe_score"]
                    ))
        else:
            # Process single image
            logger.info(f"Processing image file: {file_path}")
            pil_image = Image.open(file_path).convert("RGB")
            
            classifications = self._classify_images([pil_image], threshold)
            cls_result = classifications[0]
            
            frame_results.append(FrameResult(
                frame_index=0,
                timestamp_seconds=0.0,
                classification=cls_result["classification"],
                nsfw_score=cls_result["nsfw_score"],
                safe_score=cls_result["safe_score"]
            ))
        
        # Calculate overall results
        nsfw_frames = [fr for fr in frame_results if fr.classification == "NSFW"]
        max_nsfw_score = max(fr.nsfw_score for fr in frame_results) if frame_results else 0.0
        is_nsfw = len(nsfw_frames) > 0
        
        return AppOutput(
            is_nsfw=is_nsfw,
            overall_classification="NSFW" if is_nsfw else "SAFE",
            max_nsfw_score=max_nsfw_score,
            frames_analyzed=len(frame_results),
            nsfw_frames_count=len(nsfw_frames),
            frame_results=frame_results
        )

    async def unload(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
