import os
import sys

# Get the absolute path of the Real-ESRGAN directory
realesrgan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realesrgan')

# Add the Real-ESRGAN directory to the Python path if it's not already there
if realesrgan_dir not in sys.path:
    sys.path.insert(0, realesrgan_dir)

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from typing import Literal, Optional, Dict, List
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import cv2
import numpy as np
import tempfile
from tqdm import tqdm
from pydantic import Field

class AppInput(BaseAppInput):
    input_file: File
    model_name: Literal[
        'realesr-general-wdn-x4v3',
        'realesr-general-x4v3',
        'realesr-animevideov3',
        'RealESRGAN_x2plus',
        'RealESRGAN_x4plus_anime_6B',
        'RealESRGAN_x4plus',
    ]
    face_enhance: bool = False
    outscale: float = 4.0
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 0
    fp32: bool = False

class AppOutput(BaseAppOutput):
    result: File

class App(BaseApp):
    # Define class-level properties with proper type hints
    upsampler: Optional[RealESRGANer] = Field(default=None)
    face_enhancer: Optional[object] = Field(default=None)  # Using object since GFPGANer type might not be available at import time
    model_paths: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            'RealESRNet_x4plus': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'],
            'RealESRGAN_x4plus_anime_6B': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'],
            'RealESRGAN_x2plus': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'],
            'realesr-animevideov3': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'],
            'realesr-general-wdn-x4v3': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth'],
            'realesr-general-x4v3': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']
        }
    )

    def _choose_model(self, input: AppInput):
        """Choose and initialize the appropriate model based on input parameters."""
        half = True if torch.cuda.is_available() and not input_data.fp32 else False
        
        if input_data.model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
            scale = 4
        elif input_data.model_name == 'realesr-general-x4v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            model_path = os.path.join('weights', 'realesr-general-x4v3.pth')
            scale = 4
        elif input_data.model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model_path = os.path.join('weights', 'RealESRGAN_x4plus_anime_6B.pth')
            scale = 4
        elif input_data.model_name == 'realesr-animevideov3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            model_path = os.path.join('weights', 'realesr-animevideov3.pth')
            scale = 4
        elif input_data.model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_path = os.path.join('weights', 'RealESRGAN_x2plus.pth')
            scale = 2
        elif input_data.model_name == 'realesr-general-wdn-x4v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            model_path = os.path.join('weights', 'realesr-general-wdn-x4v3.pth')
            scale = 4
        else:
            raise ValueError(f"Unknown model name: {input_data.model_name}")
        
        # Download model if it doesn't exist
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in self.model_paths[input_data.model_name]:
                model_path = load_file_from_url(
                    url=url, 
                    model_dir=os.path.join(ROOT_DIR, 'weights'), 
                    progress=True, 
                    file_name=None
                )
        
        # Initialize upsampler
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=input_data.tile,
            tile_pad=input_data.tile_pad,
            pre_pad=input_data.pre_pad,
            half=half,
            gpu_id=0 if torch.cuda.is_available() else None
        )

    async def setup(self, metadata):
        """Initialize Real-ESRGAN models and resources."""
        # Create weights directory if it doesn't exist
        os.makedirs('weights', exist_ok=True)
        
        # Model initialization will be done per request in _choose_model method

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Run Real-ESRGAN on the input image/video."""
        # Choose and initialize the appropriate model
        self._choose_model(input)
        
        # Initialize face enhancer if needed
        if input_data.face_enhance and self.face_enhancer is None:
            from gfpgan import GFPGANer
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=input_data.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler
            )
        
        # Determine if input is video or image
        is_video = input_data.input_file.path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if is_video:
            # Handle video processing
            output_path = self._process_video(input)
        else:
            # Handle image processing
            output_path = self._process_image(input)
        
        return AppOutput(result=File(path=output_path))

    def _process_image(self, input: AppInput) -> str:
        """Process a single image using Real-ESRGAN."""
        img = cv2.imread(input_data.input_file.path, cv2.IMREAD_UNCHANGED)
        
        try:
            if input_data.face_enhance:
                _, _, output = self.face_enhancer.enhance(
                    img, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
            else:
                output, _ = self.upsampler.enhance(img, outscale=input_data.outscale)
        except RuntimeError as error:
            print('Error:', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            raise

        # Save the output
        output_path = os.path.splitext(input_data.input_file.path)[0] + '_out.png'
        cv2.imwrite(output_path, output)
        return output_path

    def _process_video(self, input: AppInput) -> str:
        """Process a video using Real-ESRGAN."""
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Read video
            video = cv2.VideoCapture(input_data.input_file.path)
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            output_path = os.path.splitext(input_data.input_file.path)[0] + '_out.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (int(width * input_data.outscale), int(height * input_data.outscale))
            )
            
            # Process each frame
            pbar = tqdm(total=total_frames, unit='frame')
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                    
                try:
                    if input_data.face_enhance:
                        _, _, output = self.face_enhancer.enhance(
                            frame,
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=True
                        )
                    else:
                        output, _ = self.upsampler.enhance(frame, outscale=input_data.outscale)
                except RuntimeError as error:
                    print('Error:', error)
                    print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                    raise
                
                # Write processed frame
                out.write(output)
                pbar.update(1)
            
            # Clean up
            pbar.close()
            video.release()
            out.release()
            
            return output_path

    async def unload(self):
        """Clean up resources."""
        self.upsampler = None
        self.face_enhancer = None
