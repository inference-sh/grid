from enum import Enum
import math
from typing import List, Tuple, Protocol, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageOps
import cv2
import numpy as np

# 1. Type Definitions & Protocols
class UpscaleFn(Protocol):
    def __call__(self, image: Image.Image, scale_factor: int) -> Image.Image: ...

class ProcessFn(Protocol):
    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image: ...
    
class UpscaleMode(str, Enum):
    LINEAR = "linear"
    CHESS = "chess"
    NONE = "none"

    @classmethod
    def _missing_(cls, value):
        # Handle integer values for backwards compatibility
        int_to_str = {
            0: cls.LINEAR,
            1: cls.CHESS,
            2: cls.NONE
        }
        if isinstance(value, int) and value in int_to_str:
            return int_to_str[value]
        return None

class SeamFixMode(str, Enum):
    NONE = "none"
    BAND_PASS = "band_pass"
    HALF_TILE = "half_tile"
    HALF_TILE_PLUS_INTERSECTIONS = "half_tile_plus_intersections"

    @classmethod
    def _missing_(cls, value):
        # Handle integer values for backwards compatibility
        int_to_str = {
            0: cls.NONE,
            1: cls.BAND_PASS,
            2: cls.HALF_TILE,
            3: cls.HALF_TILE_PLUS_INTERSECTIONS
        }
        if isinstance(value, int) and value in int_to_str:
            return int_to_str[value]
        return None

# 3. Configuration
@dataclass
class UpscaleConfig:
    target_width: int
    target_height: int
    tile_width: int = 512
    tile_height: int = 512
    upscale_mode: UpscaleMode = UpscaleMode.LINEAR
    redraw_padding: int = 32
    redraw_mask_blur: int = 8
    save_redraw: bool = True
    seam_fix_mode: SeamFixMode = SeamFixMode.NONE
    seam_fix_denoise: float = 1.0
    seam_fix_width: int = 64
    seam_fix_mask_blur: int = 8
    seam_fix_padding: int = 16
    save_seams_fix: bool = False

    def __post_init__(self):
        self.tile_width = self.tile_width if self.tile_width > 0 else self.tile_height
        self.tile_height = self.tile_height if self.tile_height > 0 else self.tile_width

# 4. Core Upscaler Class
class Upscaler:
    def __init__(self, config: UpscaleConfig):
        self.config = config

        self.rows = math.ceil(self.config.target_height / self.config.tile_height)
        self.cols = math.ceil(self.config.target_width / self.config.tile_width)
    
    def upscale(
        self,
        image: Image.Image,
        upscale_fn: UpscaleFn,
        process_fn: ProcessFn
    ) -> Image.Image:
        # Initial upscaling
        image = self._perform_initial_upscale(image, upscale_fn)
        
        # Process tiles if needed
        if self.config.upscale_mode != UpscaleMode.NONE:
            image = self.process_tiles(image, process_fn)
        
        # Fix seams if needed
        if self.config.seam_fix_mode != SeamFixMode.NONE:
            image = self.process_seams(image, process_fn)
        
        return image

    def _perform_initial_upscale(self, image: Image.Image, upscale_fn: UpscaleFn) -> Image.Image:
        factors = self._calculate_upscale_factors(
            current_size=max(image.width, image.height),
            target_size=max(self.config.target_width, self.config.target_height)
        )
        for factor in factors:
            image = upscale_fn(image, factor)
        return image.resize(
            (self.config.target_width, self.config.target_height),
            Image.LANCZOS
        )

    @staticmethod
    def _calculate_upscale_factors(current_size: int, target_size: int) -> List[int]:
        scale_factor = math.ceil(target_size / current_size)
        scales = []
        current_scale = 1

        def get_factor(num: int) -> int:
            if num == 1: return 2
            if num % 4 == 0: return 4
            if num % 3 == 0: return 3
            if num % 2 == 0: return 2
            return 0

        current_scale_factor = get_factor(scale_factor)
        while current_scale_factor == 0:
            scale_factor += 1
            current_scale_factor = get_factor(scale_factor)

        while current_scale < scale_factor:
            current_scale_factor = get_factor(scale_factor // current_scale)
            if current_scale_factor == 0:
                break
            scales.append(current_scale_factor)
            current_scale *= current_scale_factor

        return scales
    
    def create_tile_masks(self, image_width: int, image_height: int) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        masks = []

        def calc_rectangle(xi: int, yi: int) -> Tuple[int, int, int, int]:
            x1 = xi * self.config.tile_width
            y1 = yi * self.config.tile_height
            x2 = min(x1 + self.config.tile_width, image_width)
            y2 = min(y1 + self.config.tile_height, image_height)
            return x1, y1, x2, y2

        if self.config.upscale_mode == UpscaleMode.LINEAR:
            for yi in range(self.rows):
                for xi in range(self.cols):
                    mask = Image.new("L", (image_width, image_height), "black")
                    draw = ImageDraw.Draw(mask)
                    coords = calc_rectangle(xi, yi)
                    draw.rectangle(coords, fill="white")
                    masks.append((mask, coords))

        elif self.config.upscale_mode == UpscaleMode.CHESS:
            tiles = []
            for yi in range(self.rows):
                row = []
                for xi in range(self.cols):
                    color = xi % 2 == 0
                    if yi > 0 and yi % 2 != 0:
                        color = not color
                    row.append(color)
                tiles.append(row)

            for yi in range(self.rows):
                for xi in range(self.cols):
                    if tiles[yi][xi]:
                        mask = Image.new("L", (image_width, image_height), "black")
                        draw = ImageDraw.Draw(mask)
                        coords = calc_rectangle(xi, yi)
                        draw.rectangle(coords, fill="white")
                        masks.append((mask, coords))

            for yi in range(self.rows):
                for xi in range(self.cols):
                    if not tiles[yi][xi]:
                        mask = Image.new("L", (image_width, image_height), "black")
                        draw = ImageDraw.Draw(mask)
                        coords = calc_rectangle(xi, yi)
                        draw.rectangle(coords, fill="white")
                        masks.append((mask, coords))

        return masks

    def create_seam_masks(self) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        masks = []

        if self.config.seam_fix_mode == SeamFixMode.BAND_PASS:
            gradient = Image.linear_gradient("L")
            mirror_gradient = Image.new("L", (256, 256), "black")
            mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
            mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

            # Create reusable gradients first (like the reference code)
            row_gradient = mirror_gradient.resize((self.config.target_width, self.config.seam_fix_width), resample=Image.BICUBIC)
            col_gradient = mirror_gradient.rotate(90).resize((self.config.seam_fix_width, self.config.target_height), resample=Image.BICUBIC)

            # Process vertical seams
            for xi in range(1, self.cols):
                mask = Image.new("L", (self.config.target_width, self.config.target_height), "black")
                x_pos = xi * self.config.tile_width - self.config.seam_fix_width // 2
                mask.paste(col_gradient, (x_pos, 0))
                masks.append((mask, (x_pos, 0, x_pos + self.config.seam_fix_width, self.config.target_height)))

            # Process horizontal seams
            for yi in range(1, self.rows):
                mask = Image.new("L", (self.config.target_width, self.config.target_height), "black")
                y_pos = yi * self.config.tile_height - self.config.seam_fix_width // 2
                mask.paste(row_gradient, (0, y_pos))
                masks.append((mask, (0, y_pos, self.config.target_width, y_pos + self.config.seam_fix_width)))

        elif self.config.seam_fix_mode in [SeamFixMode.HALF_TILE, SeamFixMode.HALF_TILE_PLUS_INTERSECTIONS]:
            gradient = Image.linear_gradient("L")

            row_gradient = Image.new("L", (self.config.tile_width, self.config.tile_height), "black")
            row_gradient.paste(
                gradient.resize((self.config.tile_width, self.config.tile_height // 2), resample=Image.BICUBIC),
                (0, 0)
            )
            row_gradient.paste(
                gradient.rotate(180).resize((self.config.tile_width, self.config.tile_height // 2), resample=Image.BICUBIC),
                (0, self.config.tile_height // 2)
            )

            col_gradient = Image.new("L", (self.config.tile_width, self.config.tile_height), "black")
            col_gradient.paste(
                gradient.rotate(90).resize((self.config.tile_width // 2, self.config.tile_height), resample=Image.BICUBIC),
                (0, 0)
            )
            col_gradient.paste(
                gradient.rotate(270).resize((self.config.tile_width // 2, self.config.tile_height), resample=Image.BICUBIC),
                (self.config.tile_width // 2, 0)
            )

            for yi in range(self.rows - 1):
                for xi in range(self.cols):
                    mask = Image.new("L", (self.config.target_width, self.config.target_height), "black")
                    x_pos = xi * self.config.tile_width
                    y_pos = yi * self.config.tile_height + self.config.tile_height // 2
                    mask.paste(row_gradient, (x_pos, y_pos))
                    masks.append((mask, (x_pos, y_pos, x_pos + self.config.tile_width, y_pos + self.config.tile_height)))

            for yi in range(self.rows):
                for xi in range(self.cols - 1):
                    mask = Image.new("L", (self.config.target_width, self.config.target_height), "black")
                    x_pos = xi * self.config.tile_width + self.config.tile_width // 2
                    y_pos = yi * self.config.tile_height
                    mask.paste(col_gradient, (x_pos, y_pos))
                    masks.append((mask, (x_pos, y_pos, x_pos + self.config.tile_width, y_pos + self.config.tile_height)))

            if self.config.seam_fix_mode == SeamFixMode.HALF_TILE_PLUS_INTERSECTIONS:
                rg = Image.radial_gradient("L").resize((self.config.tile_width, self.config.tile_height), resample=Image.BICUBIC)
                rg = ImageOps.invert(rg)
                for yi in range(self.rows - 1):
                    for xi in range(self.cols - 1):
                        mask = Image.new("L", (self.config.target_width, self.config.target_height), "black")
                        x_pos = xi * self.config.tile_width + self.config.tile_width // 2
                        y_pos = yi * self.config.tile_height + self.config.tile_height // 2
                        mask.paste(rg, (x_pos, y_pos))
                        masks.append((mask, (x_pos, y_pos, x_pos + self.config.tile_width, y_pos + self.config.tile_height)))

        return masks
    
    def process_tile(self, image: Image.Image, mask: Image.Image, process_fn: ProcessFn, crop_region: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple[int, int]]:
        x1, y1, x2, y2 = crop_region
        tile_mask = mask.crop(crop_region)

        processing_width = math.ceil((self.config.tile_width+self.config.redraw_padding) / 64) * 64
        processing_height = math.ceil((self.config.tile_height+self.config.redraw_padding) / 64) * 64
        tile_mask = ImageUtils.resize_image(2, tile_mask, processing_width, processing_height)

        if self.config.redraw_mask_blur > 0:
            np_mask = np.array(tile_mask)
            kernel_size = 2 * int(2.5 * self.config.redraw_mask_blur + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), self.config.redraw_mask_blur)
            tile_mask = Image.fromarray(np_mask)
        
        tile_image = image.crop(crop_region)
        tile_image = ImageUtils.resize_image(2, tile_image, processing_width, processing_height)
        tile_image = process_fn(tile_image, tile_mask)
        
        cw, ch = x2 - x1, y2 - y1
        tile_image = tile_image.resize((cw, ch), Image.LANCZOS)
        tile_image = tile_image.crop((0, 0, cw, ch))
        
        return tile_image, (x1, y1)

    def process_tiles(self, image: Image.Image, process_fn: ProcessFn) -> Image.Image:
        tile_masks = self.create_tile_masks(image.width, image.height)
        for mask, (x1, y1, x2, y2) in tile_masks:
            crop_region = ImageUtils.get_crop_region(mask, self.config.redraw_padding)
            if not crop_region:
                continue
            crop_region = ImageUtils.expand_crop(
                crop_region, self.config.tile_width, self.config.tile_height, mask.width, mask.height
            )
            tile_image, paste_pos = self.process_tile(image, mask, process_fn, crop_region)
            image.paste(tile_image, paste_pos)
        return image

    def process_seams(self, image: Image.Image, process_fn: ProcessFn) -> Image.Image:
        seam_masks = self.create_seam_masks()

        processing_width = self.config.tile_width
        processing_height = self.config.tile_height

        if self.config.seam_fix_mode == SeamFixMode.BAND_PASS:
            processing_width = self.config.tile_width * 2
            processing_height = self.config.tile_height * 2
        
        processing_width = math.ceil((processing_width+self.config.seam_fix_padding) / 64) * 64
        processing_height = math.ceil((processing_height+self.config.seam_fix_padding) / 64) * 64

        for mask, (x1, y1, x2, y2) in seam_masks:
            crop_region = ImageUtils.get_crop_region(mask, self.config.seam_fix_padding)
            if not crop_region:
                continue
            crop_region = ImageUtils.expand_crop(
                crop_region, processing_width, processing_height, mask.width, mask.height
            )

            x1, y1, x2, y2 = crop_region
            mask = mask.crop(crop_region)
            seam_mask = mask
            seam_mask = ImageUtils.resize_image(2, mask, image.width, image.height)
            
            if self.config.seam_fix_mask_blur > 0 and self.config.seam_fix_mode != SeamFixMode.BAND_PASS:
                np_mask = np.array(seam_mask)
                kernel_size = 2 * int(2.5 * self.config.seam_fix_mask_blur + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), self.config.seam_fix_mask_blur)
                seam_mask = Image.fromarray(np_mask)
            
            seam_image = image.crop(crop_region)
            seam_image = ImageUtils.resize_image(2, seam_image, self.config.tile_width, self.config.tile_height)
            seam_image = process_fn(seam_image, seam_mask)
            seam_image = seam_image.resize((x2 - x1, y2 - y1), Image.LANCZOS)
            image.paste(seam_image, (x1, y1))
        return image

class ImageUtils:
    @staticmethod
    def get_crop_region(mask: Image.Image, pad: int = 0) -> Optional[Tuple[int, int, int, int]]:
        if box := mask.getbbox():
            x1, y1, x2, y2 = box
            if pad:
                return (
                    max(x1 - pad, 0), 
                    max(y1 - pad, 0),
                    min(x2 + pad, mask.size[0]),
                    min(y2 + pad, mask.size[1])
                )
            return box

    @staticmethod
    def expand_crop(crop_region: Tuple[int, int, int, int], processing_width: int, processing_height: int, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = crop_region
        ratio_crop = (x2 - x1) / (y2 - y1)
        ratio_proc = processing_width / processing_height

        if ratio_crop > ratio_proc:
            dh = (x2 - x1) / ratio_proc
            diff = int(dh - (y2 - y1))
            y1 -= diff // 2
            y2 += diff - diff // 2
            if y2 >= image_height:
                d = y2 - image_height
                y2 -= d
                y1 -= d
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if y2 >= image_height:
                y2 = image_height
        else:
            dw = (y2 - y1) * ratio_proc
            diff = int(dw - (x2 - x1))
            x1 -= diff // 2
            x2 += diff - diff // 2
            if x2 >= image_width:
                d = x2 - image_width
                x2 -= d
                x1 -= d
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if x2 >= image_width:
                x2 = image_width
        return x1, y1, x2, y2

    @staticmethod
    def resize_image(resize_mode: int, im: Image.Image, width: int, height: int, upscaler_name: str = None) -> Image.Image:
        def _resize(im, w, h):
            return im.resize((w, h), resample=Image.LANCZOS)

        if resize_mode == 0:
            return _resize(im, width, height)
        elif resize_mode == 1:
            ratio = width / height
            src_ratio = im.width / im.height
            if ratio > src_ratio:
                w = width
                h = round(im.height * (width / im.width))
            else:
                w = round(im.width * (height / im.height))
                h = height
            r = _resize(im, w, h)
            res = Image.new("RGB", (width, height))
            res.paste(r, ((width - w) // 2, (height - h) // 2))
            return res
        elif resize_mode == 2:
            ratio = width / height
            src_ratio = im.width / im.height
            if ratio > src_ratio:
                w = round(im.width * (height / im.height))
                h = height
            else:
                w = width
                h = round(im.height * (width / im.width))
            r = _resize(im, w, h)
            res = Image.new("RGB", (width, height))
            res.paste(r, ((width - w) // 2, (height - h) // 2))
            left = (width - w) // 2
            right = width - w - left
            top = (height - h) // 2
            bottom = height - h - top

            if top > 0:
                t = r.crop((0, 0, w, top))
                t = ImageOps.flip(t)
                res.paste(t, (left, 0))
            if bottom > 0:
                b = r.crop((0, h - bottom, w, h))
                b = ImageOps.flip(b)
                res.paste(b, (left, height - bottom))
            if left > 0:
                l = r.crop((0, 0, left, h))
                l = ImageOps.mirror(l)
                res.paste(l, (0, top))
            if right > 0:
                ri = r.crop((w - right, 0, w, h))
                ri = ImageOps.mirror(ri)
                res.paste(ri, (width - right, top))
            return res
        return im

# 6. Main Entry Point
def upscale(
    image: Image.Image,
    target_width: int,
    target_height: int,
    upscale_fn: UpscaleFn,
    process_fn: ProcessFn,
    # Tile configuration
    tile_width: int = 512,
    tile_height: int = 512,
    upscale_mode: UpscaleMode = UpscaleMode.LINEAR,
    # Redraw configuration
    redraw_padding: int = 32,
    redraw_mask_blur: int = 8,
    save_redraw: bool = True,
    # Seam fixing configuration
    seam_fix_mode: SeamFixMode = SeamFixMode.NONE,
    seam_fix_denoise: float = 1.0,
    seam_fix_width: int = 64,
    seam_fix_mask_blur: int = 8,
    seam_fix_padding: int = 16,
    save_seams_fix: bool = False,
) -> Image.Image:
    """Main entry point for image upscaling.
    
    Args:
        image: Input image to upscale
        target_width: Desired output width
        target_height: Desired output height
        upscale_fn: Function to perform basic upscaling
        process_fn: Function to process individual tiles
        tile_width: Width of processing tiles
        tile_height: Height of processing tiles
        upscale_mode: How to split image into tiles (LINEAR, CHESS, or NONE)
        redraw_padding: Padding around redrawn areas
        redraw_mask_blur: Blur amount for redraw masks
        save_redraw: Whether to save intermediate redraw results
        seam_fix_mode: Method to fix seams between tiles
        seam_fix_denoise: Denoising strength for seam fixing
        seam_fix_width: Width of seam fixing region
        seam_fix_mask_blur: Blur amount for seam fixing masks
        seam_fix_padding: Padding around seam fixing regions
        save_seams_fix: Whether to save intermediate seam fixing results
    
    Returns:
        Upscaled image
    """
    config = UpscaleConfig(
        target_width=target_width,
        target_height=target_height,
        tile_width=tile_width,
        tile_height=tile_height,
        upscale_mode=upscale_mode,
        redraw_padding=redraw_padding,
        redraw_mask_blur=redraw_mask_blur,
        save_redraw=save_redraw,
        seam_fix_mode=seam_fix_mode,
        seam_fix_denoise=seam_fix_denoise,
        seam_fix_width=seam_fix_width,
        seam_fix_mask_blur=seam_fix_mask_blur,
        seam_fix_padding=seam_fix_padding,
        save_seams_fix=save_seams_fix,
    )
    upscaler = Upscaler(config)
    return upscaler.upscale(image, upscale_fn, process_fn)