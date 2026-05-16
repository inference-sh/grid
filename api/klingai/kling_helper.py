"""
Kling AI API Helper Module

This module provides a unified interface for all Kling AI API endpoints including:
- Omni-Video (O1)
- Text to Video
- Image to Video
- Image Generation

API Domain: https://api-singapore.klingai.com
"""

import time
import jwt
import httpx
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

API_BASE_URL = "https://api-singapore.klingai.com"

# Video Models
class VideoModel(str, Enum):
    KLING_V1 = "kling-v1"
    KLING_V1_5 = "kling-v1-5"
    KLING_V1_6 = "kling-v1-6"
    KLING_V2_MASTER = "kling-v2-master"
    KLING_V2_1 = "kling-v2-1"
    KLING_V2_1_MASTER = "kling-v2-1-master"
    KLING_V2_5_TURBO = "kling-v2-5-turbo"
    KLING_V2_6 = "kling-v2-6"
    KLING_V3 = "kling-v3"
    KLING_V3_OMNI = "kling-v3-omni"
    KLING_VIDEO_O1 = "kling-video-o1"  # Omni model
    KLING_VIDEO_3O = "kling-video-3o"  # Video 3O model

# Image Models
class ImageModel(str, Enum):
    KLING_V1 = "kling-v1"
    KLING_V1_5 = "kling-v1-5"
    KLING_V2 = "kling-v2"
    KLING_V2_NEW = "kling-v2-new"
    KLING_V2_1 = "kling-v2-1"

# Task Status
class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    SUCCEED = "succeed"
    FAILED = "failed"

# Video Mode
class VideoMode(str, Enum):
    STD = "std"  # Standard - cost effective
    PRO = "pro"  # Professional - higher quality

# Aspect Ratios
class AspectRatio(str, Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_3_2 = "3:2"
    RATIO_2_3 = "2:3"
    RATIO_21_9 = "21:9"

# Camera Movement Types
class CameraType(str, Enum):
    SIMPLE = "simple"
    DOWN_BACK = "down_back"
    FORWARD_UP = "forward_up"
    RIGHT_TURN_FORWARD = "right_turn_forward"
    LEFT_TURN_FORWARD = "left_turn_forward"


# =============================================================================
# Error Handling
# =============================================================================

class KlingAPIError(Exception):
    """Base exception for Kling API errors"""
    def __init__(self, code: int, message: str, request_id: str = None):
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(f"[{code}] {message}")


# Error code definitions for reference
ERROR_CODES = {
    0: "Success",
    1000: "Authentication failed",
    1001: "Authorization is empty",
    1002: "Authorization is invalid",
    1003: "Authorization is not yet valid",
    1004: "Authorization has expired",
    1100: "Account exception",
    1101: "Account in arrears (postpaid)",
    1102: "Resource pack depleted or expired (prepaid)",
    1103: "Unauthorized access to requested resource",
    1200: "Invalid request parameters",
    1201: "Invalid parameters (incorrect key or illegal value)",
    1202: "The requested method is invalid",
    1203: "The requested resource does not exist",
    1300: "Trigger platform strategy",
    1301: "Trigger content security policy",
    1302: "API request rate limit exceeded",
    1303: "Concurrency/QPS exceeds prepaid limit",
    1304: "Trigger IP whitelisting policy",
    5000: "Server internal error",
    5001: "Server temporarily unavailable",
    5002: "Server internal timeout",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CameraConfig:
    """Camera movement configuration for simple type"""
    horizontal: float = 0  # [-10, 10] left/right translation
    vertical: float = 0    # [-10, 10] down/up translation
    pan: float = 0         # [-10, 10] rotation around x-axis
    tilt: float = 0        # [-10, 10] rotation around y-axis
    roll: float = 0        # [-10, 10] rotation around z-axis
    zoom: float = 0        # [-10, 10] focal length change

    def to_dict(self) -> Dict[str, float]:
        return {
            "horizontal": self.horizontal,
            "vertical": self.vertical,
            "pan": self.pan,
            "tilt": self.tilt,
            "roll": self.roll,
            "zoom": self.zoom,
        }


@dataclass
class CameraControl:
    """Camera control settings"""
    type: CameraType
    config: Optional[CameraConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type.value if isinstance(self.type, CameraType) else self.type}
        if self.config and self.type == CameraType.SIMPLE:
            result["config"] = self.config.to_dict()
        return result


@dataclass
class TrajectoryPoint:
    """A point in a motion trajectory"""
    x: int
    y: int

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass
class DynamicMask:
    """Dynamic brush mask configuration"""
    mask: str  # URL or base64
    trajectories: List[TrajectoryPoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask": self.mask,
            "trajectories": [t.to_dict() for t in self.trajectories]
        }


@dataclass
class ImageRef:
    """Image reference for Omni-Video"""
    image_url: str
    type: Optional[Literal["first_frame", "end_frame"]] = None

    def to_dict(self) -> Dict[str, str]:
        result = {"image_url": self.image_url}
        if self.type:
            result["type"] = self.type
        return result


@dataclass
class ElementRef:
    """Element reference for Omni-Video"""
    element_id: int

    def to_dict(self) -> Dict[str, int]:
        return {"element_id": self.element_id}


@dataclass
class VideoRef:
    """Video reference for Omni-Video"""
    video_url: str
    refer_type: Literal["feature", "base"] = "base"
    keep_original_sound: Literal["yes", "no"] = "yes"

    def to_dict(self) -> Dict[str, str]:
        return {
            "video_url": self.video_url,
            "refer_type": self.refer_type,
            "keep_original_sound": self.keep_original_sound,
        }


@dataclass
class WatermarkInfo:
    """Watermark configuration"""
    enabled: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {"enabled": self.enabled}


@dataclass
class VoiceRef:
    """Voice reference for video generation"""
    voice_id: str

    def to_dict(self) -> Dict[str, str]:
        return {"voice_id": self.voice_id}


@dataclass
class VideoResult:
    """Result of a video generation task"""
    id: str
    url: str
    duration: str
    watermark_url: Optional[str] = None


@dataclass
class ImageResult:
    """Result of an image generation task"""
    index: int
    url: str


@dataclass
class TaskResult:
    """Generic task result"""
    task_id: str
    task_status: TaskStatus
    task_status_msg: Optional[str] = None
    external_task_id: Optional[str] = None
    final_unit_deduction: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    videos: Optional[List[VideoResult]] = None
    images: Optional[List[ImageResult]] = None


# =============================================================================
# Main Client
# =============================================================================

class KlingClient:
    """
    Kling AI API Client

    Usage:
        client = KlingClient(access_key="your_ak", secret_key="your_sk")

        # Text to Video
        task = await client.text2video.create(
            prompt="A cat walking in the garden",
            duration="5",
            mode="pro"
        )

        # Poll for result
        result = await client.text2video.get(task.task_id)
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        base_url: str = API_BASE_URL,
        timeout: float = 30.0,
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        # Initialize sub-clients
        self.text2video = Text2VideoAPI(self)
        self.image2video = Image2VideoAPI(self)
        self.omni_video = OmniVideoAPI(self)
        self.images = ImageGenerationAPI(self)
        self.lip_sync = LipSyncAPI(self)
        self.avatar = AvatarAPI(self)
        self.video_to_audio = VideoToAudioAPI(self)
        self.virtual_tryon = VirtualTryOnAPI(self)
        self.omni_image = OmniImageAPI(self)

    def _generate_token(self) -> str:
        """Generate JWT token for API authentication"""
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,  # Valid for 30 minutes
            "nbf": int(time.time()) - 5      # Valid from 5 seconds ago
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256", headers=headers)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authorization"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._generate_token()}"
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an API request"""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        response = await client.request(
            method=method,
            url=url,
            headers=self._get_headers(),
            json=json_data,
            params=params,
        )

        data = response.json()

        if data.get("code", 0) != 0:
            raise KlingAPIError(
                code=data.get("code"),
                message=data.get("message", "Unknown error"),
                request_id=data.get("request_id"),
            )

        return data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# =============================================================================
# Text to Video API
# =============================================================================

class Text2VideoAPI:
    """Text to Video API endpoints"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def create(
        self,
        prompt: str,
        model_name: str = "kling-v1",
        negative_prompt: Optional[str] = None,
        sound: Literal["on", "off"] = "off",
        cfg_scale: float = 0.5,
        mode: str = "std",
        camera_control: Optional[CameraControl] = None,
        aspect_ratio: str = "16:9",
        duration: str = "5",
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create a text-to-video generation task.

        Args:
            prompt: Text description of the video (max 2500 chars)
            model_name: Model version (kling-v1, kling-v1-6, kling-v2-master, etc.)
            negative_prompt: What to avoid in the video (max 2500 chars)
            sound: Generate sound (on/off) - only V2.6+ supports this
            cfg_scale: Flexibility [0,1] - higher = stricter adherence to prompt
            mode: Generation mode (std=standard, pro=professional)
            camera_control: Camera movement settings
            aspect_ratio: Video aspect ratio (16:9, 9:16, 1:1)
            duration: Video length in seconds (5 or 10)
            callback_url: Webhook URL for task status updates
            external_task_id: Custom task ID (must be unique per account)

        Returns:
            TaskResult with task_id and initial status
        """
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if sound != "off":
            payload["sound"] = sound
        if cfg_scale != 0.5:
            payload["cfg_scale"] = cfg_scale
        if camera_control:
            payload["camera_control"] = camera_control.to_dict()
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/videos/text2video", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        """Get status and result of a text-to-video task"""
        data = await self._client._request("GET", f"/v1/videos/text2video/{task_id}")
        return self._parse_task_result(data)

    async def get_by_external_id(self, external_task_id: str) -> TaskResult:
        """Get task by custom external ID"""
        data = await self._client._request("GET", f"/v1/videos/text2video/{external_task_id}")
        return self._parse_task_result(data)

    async def list(self, page_num: int = 1, page_size: int = 30) -> List[TaskResult]:
        """List text-to-video tasks with pagination"""
        data = await self._client._request(
            "GET",
            "/v1/videos/text2video",
            params={"pageNum": page_num, "pageSize": page_size}
        )
        return [self._parse_task_result({"data": item}) for item in data.get("data", [])]

    def _parse_task_result(self, data: Dict) -> TaskResult:
        """Parse API response into TaskResult"""
        task_data = data.get("data", {})

        videos = None
        if "task_result" in task_data and "videos" in task_data["task_result"]:
            videos = [
                VideoResult(
                    id=v.get("id", ""),
                    url=v.get("url", ""),
                    duration=v.get("duration", ""),
                    watermark_url=v.get("watermark_url"),
                )
                for v in task_data["task_result"]["videos"]
            ]

        task_info = task_data.get("task_info", {})

        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            videos=videos,
        )


# =============================================================================
# Image to Video API
# =============================================================================

class Image2VideoAPI:
    """Image to Video API endpoints"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def create(
        self,
        image: str,
        prompt: Optional[str] = None,
        model_name: str = "kling-v1",
        image_tail: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        voice_list: Optional[List[VoiceRef]] = None,
        sound: Literal["on", "off"] = "off",
        cfg_scale: float = 0.5,
        mode: str = "std",
        static_mask: Optional[str] = None,
        dynamic_masks: Optional[List[DynamicMask]] = None,
        camera_control: Optional[CameraControl] = None,
        duration: str = "5",
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create an image-to-video generation task.

        Args:
            image: Reference image (URL or base64, no data: prefix)
            prompt: Text description (max 2500 chars). Use <<<voice_1>>> for voice refs
            model_name: Model version
            image_tail: End frame image (URL or base64)
            negative_prompt: What to avoid (max 2500 chars)
            voice_list: Voice references for speech (max 2)
            sound: Generate sound (on/off) - V2.6+ only
            cfg_scale: Flexibility [0,1]
            mode: Generation mode (std/pro)
            static_mask: Static brush mask (URL or base64)
            dynamic_masks: Dynamic brush configurations (max 6 groups)
            camera_control: Camera movement settings
            duration: Video length (5 or 10 seconds)
            callback_url: Webhook URL
            external_task_id: Custom task ID

        Returns:
            TaskResult with task_id
        """
        payload = {
            "model_name": model_name,
            "image": image,
            "mode": mode,
            "duration": duration,
        }

        if prompt:
            payload["prompt"] = prompt
        if image_tail:
            payload["image_tail"] = image_tail
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if voice_list:
            payload["voice_list"] = [v.to_dict() for v in voice_list]
        if sound != "off":
            payload["sound"] = sound
        if cfg_scale != 0.5:
            payload["cfg_scale"] = cfg_scale
        if static_mask:
            payload["static_mask"] = static_mask
        if dynamic_masks:
            payload["dynamic_masks"] = [m.to_dict() for m in dynamic_masks]
        if camera_control:
            payload["camera_control"] = camera_control.to_dict()
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/videos/image2video", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        """Get status and result of an image-to-video task"""
        data = await self._client._request("GET", f"/v1/videos/image2video/{task_id}")
        return self._parse_task_result(data)

    async def list(self, page_num: int = 1, page_size: int = 30) -> List[TaskResult]:
        """List image-to-video tasks"""
        data = await self._client._request(
            "GET",
            "/v1/videos/image2video",
            params={"pageNum": page_num, "pageSize": page_size}
        )
        return [self._parse_task_result({"data": item}) for item in data.get("data", [])]

    def _parse_task_result(self, data: Dict) -> TaskResult:
        """Parse API response into TaskResult"""
        task_data = data.get("data", {})

        videos = None
        if "task_result" in task_data and "videos" in task_data["task_result"]:
            videos = [
                VideoResult(
                    id=v.get("id", ""),
                    url=v.get("url", ""),
                    duration=v.get("duration", ""),
                )
                for v in task_data["task_result"]["videos"]
            ]

        task_info = task_data.get("task_info", {})

        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            videos=videos,
        )


# =============================================================================
# Omni-Video API
# =============================================================================

class OmniVideoAPI:
    """Omni-Video (O1) API endpoints - unified video generation"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def create(
        self,
        prompt: str,
        model_name: str = "kling-video-o1",
        image_list: Optional[List[ImageRef]] = None,
        element_list: Optional[List[ElementRef]] = None,
        video_list: Optional[List[VideoRef]] = None,
        mode: str = "pro",
        aspect_ratio: Optional[str] = None,
        duration: str = "5",
        watermark_info: Optional[WatermarkInfo] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create an Omni-Video generation task.

        The Omni model supports various capabilities through templated prompts:
        - Reference elements: <<<element_1>>>, <<<element_2>>>
        - Reference images: <<<image_1>>>, <<<image_2>>>
        - Reference videos: <<<video_1>>>

        Args:
            prompt: Text prompt with optional references (max 2500 chars)
            model_name: Must be "kling-video-o1"
            image_list: Reference images (for elements, scenes, styles, first/last frames)
            element_list: Reference elements by ID
            video_list: Reference videos (for editing or feature reference)
            mode: Generation mode (std/pro)
            aspect_ratio: Required when not using first-frame or video editing
            duration: Video length (3-10 seconds depending on use case)
            watermark_info: Watermark settings
            callback_url: Webhook URL
            external_task_id: Custom task ID

        Returns:
            TaskResult with task_id

        Notes:
            - With reference video: max 4 images + elements combined
            - Without reference video: max 7 images + elements combined
            - End frame requires first frame
            - Video editing (refer_type=base) can't use first/last frame
        """
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "mode": mode,
            "duration": duration,
        }

        if image_list:
            payload["image_list"] = [img.to_dict() for img in image_list]
        if element_list:
            payload["element_list"] = [elem.to_dict() for elem in element_list]
        if video_list:
            payload["video_list"] = [vid.to_dict() for vid in video_list]
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if watermark_info:
            payload["watermark_info"] = watermark_info.to_dict()
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/videos/omni-video", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        """Get status and result of an Omni-Video task"""
        data = await self._client._request("GET", f"/v1/videos/omni-video/{task_id}")
        return self._parse_task_result(data)

    async def list(self, page_num: int = 1, page_size: int = 30) -> List[TaskResult]:
        """List Omni-Video tasks"""
        data = await self._client._request(
            "GET",
            "/v1/videos/omni-video",
            params={"pageNum": page_num, "pageSize": page_size}
        )
        return [self._parse_task_result({"data": item}) for item in data.get("data", [])]

    def _parse_task_result(self, data: Dict) -> TaskResult:
        """Parse API response into TaskResult"""
        task_data = data.get("data", {})

        videos = None
        if "task_result" in task_data and "videos" in task_data["task_result"]:
            videos = [
                VideoResult(
                    id=v.get("id", ""),
                    url=v.get("url", ""),
                    duration=v.get("duration", ""),
                    watermark_url=v.get("watermark_url"),
                )
                for v in task_data["task_result"]["videos"]
            ]

        task_info = task_data.get("task_info", {})

        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            videos=videos,
        )


# =============================================================================
# Image Generation API
# =============================================================================

class ImageGenerationAPI:
    """Image Generation API endpoints"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def create(
        self,
        prompt: str,
        model_name: str = "kling-v1",
        negative_prompt: Optional[str] = None,
        image: Optional[str] = None,
        image_reference: Optional[Literal["subject", "face"]] = None,
        image_fidelity: float = 0.5,
        human_fidelity: float = 0.45,
        resolution: Literal["1k", "2k"] = "1k",
        n: int = 1,
        aspect_ratio: str = "16:9",
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create an image generation task.

        Args:
            prompt: Text description (max 2500 chars)
            model_name: Model version (kling-v1, kling-v1-5, kling-v2, etc.)
            negative_prompt: What to avoid (not supported with image reference)
            image: Reference image (URL or base64, required if image_reference is set)
            image_reference: Reference type (subject=character features, face=appearance)
            image_fidelity: Reference intensity [0,1]
            human_fidelity: Facial similarity [0,1] (only for subject reference)
            resolution: Output resolution (1k or 2k)
            n: Number of images to generate [1,9]
            aspect_ratio: Output aspect ratio
            callback_url: Webhook URL
            external_task_id: Custom task ID

        Returns:
            TaskResult with task_id
        """
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "resolution": resolution,
            "n": n,
            "aspect_ratio": aspect_ratio,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if image:
            payload["image"] = image
        if image_reference:
            payload["image_reference"] = image_reference
        if image_fidelity != 0.5:
            payload["image_fidelity"] = image_fidelity
        if human_fidelity != 0.45:
            payload["human_fidelity"] = human_fidelity
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/images/generations", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        """Get status and result of an image generation task"""
        data = await self._client._request("GET", f"/v1/images/generations/{task_id}")
        return self._parse_task_result(data)

    async def list(self, page_num: int = 1, page_size: int = 30) -> List[TaskResult]:
        """List image generation tasks"""
        data = await self._client._request(
            "GET",
            "/v1/images/generations",
            params={"pageNum": page_num, "pageSize": page_size}
        )
        return [self._parse_task_result({"data": item}) for item in data.get("data", [])]

    def _parse_task_result(self, data: Dict) -> TaskResult:
        """Parse API response into TaskResult"""
        task_data = data.get("data", {})

        images = None
        if "task_result" in task_data and "images" in task_data["task_result"]:
            images = [
                ImageResult(
                    index=img.get("index", 0),
                    url=img.get("url", ""),
                )
                for img in task_data["task_result"]["images"]
            ]

        task_info = task_data.get("task_info", {})

        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            images=images,
        )


# =============================================================================
# Lip Sync API
# =============================================================================

class LipSyncAPI:
    """Lip Sync API endpoints (advanced lip-sync with face recognition)"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def identify_face(self, video_url: str) -> Dict[str, Any]:
        """
        Identify faces in a video. Returns session_id and face data.
        Must be called before create() to get session_id and face_id.
        """
        data = await self._client._request(
            "POST", "/v1/videos/identify-face",
            json_data={"video_url": video_url},
        )
        return data.get("data", {})

    async def create(
        self,
        session_id: str,
        face_id: str,
        sound_file: Optional[str] = None,
        audio_id: Optional[str] = None,
        sound_start_time: int = 0,
        sound_end_time: Optional[int] = None,
        sound_insert_time: int = 0,
        sound_volume: int = 2,
        original_audio_volume: int = 2,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create an advanced lip sync task.

        Requires session_id from identify_face() call.
        Either audio_id or sound_file must be provided.

        Args:
            session_id: From identify_face response
            face_id: Face ID from identify_face response
            sound_file: Audio URL or base64 (mp3/wav/m4a/aac, max 5MB, 2-60s)
            audio_id: TTS audio ID
            sound_start_time: Audio crop start (ms, default 0)
            sound_end_time: Audio crop end (ms)
            sound_insert_time: When to insert audio in video (ms)
            sound_volume: Audio volume (0-10, default 2)
            original_audio_volume: Original video audio volume (0-10, default 2)
        """
        face_config = {
            "face_id": face_id,
            "sound_insert_time": sound_insert_time,
            "sound_start_time": sound_start_time,
            "sound_volume": sound_volume,
            "original_audio_volume": original_audio_volume,
        }
        if sound_end_time is not None:
            face_config["sound_end_time"] = sound_end_time
        if sound_file:
            face_config["sound_file"] = sound_file
        if audio_id:
            face_config["audio_id"] = audio_id

        payload = {
            "session_id": session_id,
            "face_choose": [face_config],
        }
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/videos/advanced-lip-sync", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        data = await self._client._request("GET", f"/v1/videos/advanced-lip-sync/{task_id}")
        return self._parse_task_result(data)

    def _parse_task_result(self, data: Dict) -> TaskResult:
        task_data = data.get("data", {})
        videos = None
        if "task_result" in task_data and "videos" in task_data["task_result"]:
            videos = [
                VideoResult(
                    id=v.get("id", ""),
                    url=v.get("url", ""),
                    duration=v.get("duration", ""),
                )
                for v in task_data["task_result"]["videos"]
            ]
        task_info = task_data.get("task_info", {})
        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            videos=videos,
        )


# =============================================================================
# Avatar API
# =============================================================================

class AvatarAPI:
    """Avatar (digital human) API endpoints"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def create(
        self,
        image: str,
        sound_file: Optional[str] = None,
        audio_id: Optional[str] = None,
        prompt: Optional[str] = None,
        mode: str = "std",
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create an avatar video task.

        Either audio_id OR sound_file must be provided.

        Args:
            image: Face image (URL or base64, no data: prefix)
            sound_file: Audio URL or base64 (mp3/wav/m4a/aac, max 5MB, 2-300s)
            audio_id: TTS audio ID (from Kling TTS API)
            prompt: Action/emotion prompt for avatar (max 2500 chars)
            mode: std or pro
        """
        payload = {"image": image, "mode": mode}

        if sound_file:
            payload["sound_file"] = sound_file
        if audio_id:
            payload["audio_id"] = audio_id
        if prompt:
            payload["prompt"] = prompt
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/videos/avatar/image2video", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        data = await self._client._request("GET", f"/v1/videos/avatar/image2video/{task_id}")
        return self._parse_task_result(data)

    def _parse_task_result(self, data: Dict) -> TaskResult:
        task_data = data.get("data", {})
        videos = None
        if "task_result" in task_data and "videos" in task_data["task_result"]:
            videos = [
                VideoResult(
                    id=v.get("id", ""),
                    url=v.get("url", ""),
                    duration=v.get("duration", ""),
                )
                for v in task_data["task_result"]["videos"]
            ]
        task_info = task_data.get("task_info", {})
        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            videos=videos,
        )


# =============================================================================
# Video to Audio API
# =============================================================================

class VideoToAudioAPI:
    """Video to Audio API endpoints"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def create(
        self,
        video_url: Optional[str] = None,
        video_id: Optional[str] = None,
        sound_effect_prompt: Optional[str] = None,
        bgm_prompt: Optional[str] = None,
        asmr_mode: bool = False,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create a video-to-audio task. Adds sound to a video.

        Args:
            video_url: Video URL (3-20s, mp4/mov, max 100MB)
            video_id: Kling video ID (alternative to video_url)
            sound_effect_prompt: Sound effect description (max 200 chars)
            bgm_prompt: Background music description (max 200 chars)
            asmr_mode: Enable ASMR mode for enhanced detail sounds
        """
        payload = {}

        if video_url:
            payload["video_url"] = video_url
        if video_id:
            payload["video_id"] = video_id
        if sound_effect_prompt:
            payload["sound_effect_prompt"] = sound_effect_prompt
        if bgm_prompt:
            payload["bgm_prompt"] = bgm_prompt
        if asmr_mode:
            payload["asmr_mode"] = True
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/audio/video-to-audio", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        data = await self._client._request("GET", f"/v1/audio/video-to-audio/{task_id}")
        return self._parse_task_result(data)

    def _parse_task_result(self, data: Dict) -> TaskResult:
        task_data = data.get("data", {})
        videos = None
        if "task_result" in task_data and "videos" in task_data["task_result"]:
            videos = [
                VideoResult(
                    id=v.get("id", ""),
                    url=v.get("url", ""),
                    duration=v.get("duration", ""),
                )
                for v in task_data["task_result"]["videos"]
            ]
        task_info = task_data.get("task_info", {})
        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            videos=videos,
        )


# =============================================================================
# Omni Image API
# =============================================================================

class OmniImageAPI:
    """Omni Image API - for kling-image-o1 and kling-v3-omni models"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def create(
        self,
        prompt: str,
        model_name: str = "kling-image-o1",
        image_list: Optional[List[Dict[str, str]]] = None,
        element_list: Optional[List[Dict[str, int]]] = None,
        resolution: str = "1k",
        result_type: str = "single",
        n: int = 1,
        aspect_ratio: Optional[str] = None,
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create an omni-image generation task.

        Args:
            prompt: Text prompt (max 2500 chars). Use <<<image_1>>> for refs.
            model_name: kling-image-o1 or kling-v3-omni
            image_list: Reference images [{"image": "url"}]
            element_list: Element refs [{"element_id": 123}]
            resolution: 1k, 2k, or 4k (4k only for kling-v3-omni)
            result_type: single or series (series only for kling-v3-omni)
            n: Number of images [1-9]
            aspect_ratio: Output ratio
        """
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "resolution": resolution,
            "n": n,
        }

        if image_list:
            payload["image_list"] = image_list
        if element_list:
            payload["element_list"] = element_list
        if result_type != "single":
            payload["result_type"] = result_type
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/images/omni-image", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        data = await self._client._request("GET", f"/v1/images/omni-image/{task_id}")
        return self._parse_task_result(data)

    def _parse_task_result(self, data: Dict) -> TaskResult:
        task_data = data.get("data", {})
        images = None
        if "task_result" in task_data and "images" in task_data["task_result"]:
            images = [
                ImageResult(
                    index=img.get("index", 0),
                    url=img.get("url", ""),
                )
                for img in task_data["task_result"]["images"]
            ]
        task_info = task_data.get("task_info", {})
        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            images=images,
        )


# =============================================================================
# Virtual Try-On API
# =============================================================================

class VirtualTryOnAPI:
    """Virtual Try-On API endpoints"""

    def __init__(self, client: KlingClient):
        self._client = client

    async def create(
        self,
        human_image: str,
        cloth_image: str,
        model_name: str = "kolors-virtual-try-on-v1-5",
        callback_url: Optional[str] = None,
        external_task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Create a virtual try-on task.

        Args:
            human_image: Person image (URL or base64, no data: prefix)
            cloth_image: Clothing image (URL or base64, product photo or white bg)
            model_name: kolors-virtual-try-on-v1 or kolors-virtual-try-on-v1-5
        """
        payload = {
            "model_name": model_name,
            "human_image": human_image,
            "cloth_image": cloth_image,
        }

        if callback_url:
            payload["callback_url"] = callback_url
        if external_task_id:
            payload["external_task_id"] = external_task_id

        data = await self._client._request("POST", "/v1/images/kolors-virtual-try-on", json_data=payload)
        return self._parse_task_result(data)

    async def get(self, task_id: str) -> TaskResult:
        data = await self._client._request("GET", f"/v1/images/kolors-virtual-try-on/{task_id}")
        return self._parse_task_result(data)

    def _parse_task_result(self, data: Dict) -> TaskResult:
        task_data = data.get("data", {})
        images = None
        if "task_result" in task_data and "images" in task_data["task_result"]:
            images = [
                ImageResult(
                    index=img.get("index", 0),
                    url=img.get("url", ""),
                )
                for img in task_data["task_result"]["images"]
            ]
        task_info = task_data.get("task_info", {})
        return TaskResult(
            task_id=task_data.get("task_id", ""),
            task_status=TaskStatus(task_data.get("task_status", "submitted")),
            task_status_msg=task_data.get("task_status_msg"),
            external_task_id=task_info.get("external_task_id"),
            final_unit_deduction=task_data.get("final_unit_deduction"),
            created_at=task_data.get("created_at"),
            updated_at=task_data.get("updated_at"),
            images=images,
        )


# =============================================================================
# Utility Functions
# =============================================================================

async def poll_task(
    get_func,
    task_id: str,
    interval: float = 5.0,
    timeout: float = 600.0,
) -> TaskResult:
    """
    Poll a task until completion.

    Args:
        get_func: The get method to call (e.g., client.text2video.get)
        task_id: Task ID to poll
        interval: Seconds between polls
        timeout: Maximum seconds to wait

    Returns:
        Final TaskResult

    Raises:
        TimeoutError: If task doesn't complete within timeout
        KlingAPIError: If task fails
    """
    import asyncio

    start = time.time()
    while time.time() - start < timeout:
        result = await get_func(task_id)

        if result.task_status == TaskStatus.SUCCEED:
            return result
        elif result.task_status == TaskStatus.FAILED:
            raise KlingAPIError(
                code=-1,
                message=f"Task failed: {result.task_status_msg or 'Unknown error'}",
            )

        await asyncio.sleep(interval)

    raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")


def validate_image_input(image: str) -> bool:
    """
    Validate image input format.

    Args:
        image: Image URL or base64 string

    Returns:
        True if valid

    Note:
        Base64 should NOT have data:image/... prefix
    """
    if image.startswith("data:"):
        return False  # Should not have data: prefix
    if image.startswith("http://") or image.startswith("https://"):
        return True  # Valid URL
    # Assume base64 - basic validation
    try:
        import base64
        base64.b64decode(image[:100])  # Just check start
        return True
    except Exception:
        return False
