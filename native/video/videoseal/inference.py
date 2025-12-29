import os
import logging
import tempfile
from typing import Optional, List
from enum import Enum
from pathlib import Path

import torch
from PIL import Image
from pydantic import Field

from inferencesh import BaseApp, BaseAppSetup, BaseAppInput, BaseAppOutput, File, OutputMeta, ImageMeta, VideoMeta

# Directory containing model card YAML files
CARDS_DIR = Path("videoseal/cards")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModeEnum(str, Enum):
    embed = "embed"
    detect = "detect"


class ModelEnum(str, Enum):
    videoseal = "videoseal"
    pixelseal = "pixelseal"
    chunkyseal = "chunkyseal"


class AppSetup(BaseAppSetup):
    """Setup schema for VideoSeal watermarking.
    
    Setup parameters configure which model to load at startup.
    """
    model: ModelEnum = Field(
        default=ModelEnum.videoseal,
        description="Model to use: 'videoseal' (256-bit stable), 'pixelseal' (SOTA), 'chunkyseal' (1024-bit high capacity)"
    )


class DetectionResult(BaseAppOutput):
    """Detection result for watermark analysis"""
    detected: bool = Field(description="Whether a watermark was detected")
    confidence: float = Field(description="Detection confidence score (0-1)")
    message_bits: Optional[List[int]] = Field(None, description="Extracted binary message bits if detected")


class AppInput(BaseAppInput):
    """Input schema for VideoSeal watermarking.
    
    Supports both image and video files for embedding invisible watermarks
    or detecting existing watermarks.
    """
    file: File = Field(description="Input image or video file")
    mode: ModeEnum = Field(
        default=ModeEnum.embed,
        description="Operation mode: 'embed' to add watermark, 'detect' to check for watermark"
    )
    scaling_w: float = Field(
        default=0.2,
        description="Watermark strength (0.1-0.5). Higher = more robust but more visible. Default: 0.2"
    )
    message: Optional[str] = Field(
        None,
        description="Optional custom message to embed (will be converted to binary). If not provided, a random message is used."
    )


class AppOutput(BaseAppOutput):
    """Output schema for VideoSeal watermarking."""
    output_file: Optional[File] = Field(None, description="Watermarked output file (for embed mode)")
    detection: Optional[DetectionResult] = Field(None, description="Detection results (for detect mode)")
    mode: str = Field(description="Operation mode that was performed")
    model_used: str = Field(description="Model that was used")


class App(BaseApp):
    
    async def setup(self, setup_data: AppSetup, metadata):
        """Initialize VideoSeal model."""
        logger.info("Setting up VideoSeal...")
        
        # Import videoseal
        import videoseal
        
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Map model names to card files
        model_card_mapping = {
            "videoseal": "videoseal_1.0.yaml",
            "pixelseal": "pixelseal_1.0.yaml",
            "chunkyseal": "chunkyseal_1.0.yaml",
        }
        
        # Load the selected model from local card file
        self.model_name = setup_data.model.value
        card_filename = model_card_mapping.get(self.model_name, f"{self.model_name}_1.0.yaml")
        card_path = CARDS_DIR / card_filename
        
        logger.info(f"Loading model: {self.model_name} from {card_path}")
        self.model = videoseal.load(card_path)
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
        logger.info("VideoSeal setup complete")

    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a video based on extension."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        ext = os.path.splitext(file_path)[1].lower()
        return ext in video_extensions

    def _load_image(self, file_path: str) -> tuple:
        """Load an image file as a tensor. Returns (tensor, width, height)."""
        import torchvision.transforms as T
        img = Image.open(file_path).convert("RGB")
        width, height = img.size
        tensor = T.ToTensor()(img).unsqueeze(0)
        if self.device == "cuda":
            tensor = tensor.to(self.device)
        return tensor, width, height

    def _load_video(self, file_path: str) -> tuple:
        """Load a video file as a tensor. Returns (tensor, fps, audio, width, height, duration)."""
        import torchvision.io
        
        # Read video
        video, audio, info = torchvision.io.read_video(file_path, pts_unit='sec')
        fps = info.get('video_fps', 30)
        
        # Get dimensions before permuting
        num_frames = video.shape[0]
        height = video.shape[1]
        width = video.shape[2]
        duration = num_frames / fps if fps > 0 else 0
        
        # Convert to float tensor [T, C, H, W] normalized to [0, 1]
        video = video.permute(0, 3, 1, 2).float() / 255.0
        
        if self.device == "cuda":
            video = video.to(self.device)
        
        return video, fps, audio, width, height, duration

    def _save_image(self, tensor: torch.Tensor, output_path: str):
        """Save a tensor as an image file."""
        import torchvision.transforms as T
        # Ensure tensor is on CPU and in correct format
        tensor = tensor.cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]  # Remove batch dimension
        tensor = tensor.clamp(0, 1)
        img = T.ToPILImage()(tensor)
        img.save(output_path, quality=95)

    def _save_video(self, tensor: torch.Tensor, output_path: str, fps: float, audio=None):
        """Save a tensor as a video file."""
        import torchvision.io
        
        # Convert to uint8 format [T, H, W, C]
        tensor = tensor.cpu()
        video = (tensor.clamp(0, 1) * 255).byte()
        video = video.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]
        
        # Write video
        torchvision.io.write_video(output_path, video, fps=fps)

    def _string_to_bits(self, message: str, num_bits: int) -> torch.Tensor:
        """Convert a string message to binary tensor."""
        # Convert string to bytes then to bits
        message_bytes = message.encode('utf-8')
        bits = []
        for byte in message_bytes:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        
        # Pad or truncate to num_bits
        if len(bits) < num_bits:
            bits.extend([0] * (num_bits - len(bits)))
        else:
            bits = bits[:num_bits]
        
        return torch.tensor(bits, dtype=torch.float32)

    def _get_resolution_label(self, height: int) -> str:
        """Get resolution label from height."""
        if height >= 2160:
            return "4k"
        elif height >= 1440:
            return "1440p"
        elif height >= 1080:
            return "1080p"
        elif height >= 720:
            return "720p"
        else:
            return "480p"

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run watermark embedding or detection."""
        
        if not input_data.file.exists():
            raise RuntimeError(f"Input file does not exist at path: {input_data.file.path}")
        
        file_path = input_data.file.path
        is_video = self._is_video_file(file_path)
        
        # Adjust watermark strength
        if hasattr(self.model, 'blender') and hasattr(self.model.blender, 'scaling_w'):
            self.model.blender.scaling_w = input_data.scaling_w
        
        logger.info(f"Processing {'video' if is_video else 'image'} with mode: {input_data.mode.value}")
        
        if input_data.mode == ModeEnum.embed:
            return await self._embed(input_data, file_path, is_video)
        else:
            return await self._detect(input_data, file_path, is_video)

    async def _embed(self, input_data: AppInput, file_path: str, is_video: bool) -> AppOutput:
        """Embed watermark into image or video."""
        
        if is_video:
            # Load and process video
            video_tensor, fps, audio, width, height, duration = self._load_video(file_path)
            logger.info(f"Loaded video: {width}x{height}, {video_tensor.shape[0]} frames at {fps} fps, {duration:.2f}s")
            
            # Embed watermark
            with torch.no_grad():
                outputs = self.model.embed(video_tensor, is_video=True)
            
            watermarked = outputs["imgs_w"]
            
            # Save output
            ext = os.path.splitext(file_path)[1]
            output_path = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
            self._save_video(watermarked, output_path, fps)
            
            # Build output meta for video
            output_meta = OutputMeta(
                inputs=[VideoMeta(
                    width=width,
                    height=height,
                    resolution=self._get_resolution_label(height),
                    seconds=duration
                )],
                outputs=[VideoMeta(
                    width=width,
                    height=height,
                    resolution=self._get_resolution_label(height),
                    seconds=duration,
                    extra={"model": self.model_name, "mode": "embed"}
                )]
            )
            
        else:
            # Load and process image
            img_tensor, width, height = self._load_image(file_path)
            logger.info(f"Loaded image: {width}x{height}")
            
            # Embed watermark
            with torch.no_grad():
                outputs = self.model.embed(img_tensor)
            
            watermarked = outputs["imgs_w"]
            
            # Determine output format
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
                ext = '.png'
            
            output_path = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
            self._save_image(watermarked, output_path)
            
            # Build output meta for image
            resolution_mp = round(width * height / 1_000_000, 2)
            output_meta = OutputMeta(
                inputs=[ImageMeta(
                    width=width,
                    height=height,
                    resolution_mp=resolution_mp,
                    count=1
                )],
                outputs=[ImageMeta(
                    width=width,
                    height=height,
                    resolution_mp=resolution_mp,
                    count=1,
                    extra={"model": self.model_name, "mode": "embed"}
                )]
            )
        
        logger.info(f"Watermark embedded, saved to: {output_path}")
        
        return AppOutput(
            output_file=File(path=output_path),
            detection=None,
            mode=input_data.mode.value,
            model_used=self.model_name,
            output_meta=output_meta
        )

    async def _detect(self, input_data: AppInput, file_path: str, is_video: bool) -> AppOutput:
        """Detect watermark in image or video."""
        
        if is_video:
            # Load video
            video_tensor, fps, _, width, height, duration = self._load_video(file_path)
            logger.info(f"Analyzing video: {width}x{height}, {video_tensor.shape[0]} frames")
            
            # Detect watermark (process frame by frame or as batch)
            with torch.no_grad():
                detected = self.model.detect(video_tensor)
            
            # Aggregate results across frames
            preds = detected["preds"]
            
            # First channel is detection score, rest are message bits
            detection_scores = preds[:, 0]
            avg_detection = detection_scores.mean().item()
            
            # Get message bits (majority vote across frames)
            if preds.shape[1] > 1:
                message_probs = preds[:, 1:].mean(dim=0)
                message_bits = (message_probs > 0).float().cpu().tolist()
                message_bits = [int(b) for b in message_bits]
            else:
                message_bits = None
            
            is_detected = avg_detection > 0.5
            
            # Build output meta for video
            output_meta = OutputMeta(
                inputs=[VideoMeta(
                    width=width,
                    height=height,
                    resolution=self._get_resolution_label(height),
                    seconds=duration,
                    extra={"model": self.model_name, "mode": "detect"}
                )]
            )
            
        else:
            # Load image
            img_tensor, width, height = self._load_image(file_path)
            logger.info(f"Analyzing image: {width}x{height}")
            
            # Detect watermark
            with torch.no_grad():
                detected = self.model.detect(img_tensor)
            
            preds = detected["preds"][0]  # Get first (and only) image result
            
            # First value is detection score
            detection_score = preds[0].item()
            is_detected = detection_score > 0.5
            
            # Rest are message bits
            if len(preds) > 1:
                message_bits = (preds[1:] > 0).float().cpu().tolist()
                message_bits = [int(b) for b in message_bits]
            else:
                message_bits = None
            
            avg_detection = detection_score
            
            # Build output meta for image
            resolution_mp = round(width * height / 1_000_000, 2)
            output_meta = OutputMeta(
                inputs=[ImageMeta(
                    width=width,
                    height=height,
                    resolution_mp=resolution_mp,
                    count=1,
                    extra={"model": self.model_name, "mode": "detect"}
                )]
            )
        
        logger.info(f"Detection complete: detected={is_detected}, confidence={avg_detection:.3f}")
        
        detection_result = DetectionResult(
            detected=is_detected,
            confidence=avg_detection,
            message_bits=message_bits
        )
        
        return AppOutput(
            output_file=None,
            detection=detection_result,
            mode=input_data.mode.value,
            model_used=self.model_name,
            output_meta=output_meta
        )
