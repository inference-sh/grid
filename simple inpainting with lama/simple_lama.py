import os
import torch
import cv2
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import argparse

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def prepare_img_and_mask(image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray, device: torch.device, pad_out_to_modulo: int = 8, scale_factor: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    def get_image(img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            img = img.copy()
        else:
            raise TypeError("Input image should be either PIL Image or numpy array!")
        
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # chw
        elif img.ndim == 2:
            img = img[np.newaxis, ...]
        
        assert img.ndim == 3, "Image must have 3 dimensions"
        return img.astype(np.float32) / 255

    def scale_image(img, factor, interpolation=cv2.INTER_AREA):
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))
        
        img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)
        return img[None, ...] if img.ndim == 2 else np.transpose(img, (2, 0, 1))

    def pad_img_to_modulo(img, mod):
        channels, height, width = img.shape
        out_height = height if height % mod == 0 else ((height // mod + 1) * mod)
        out_width = width if width % mod == 0 else ((width // mod + 1) * mod)
        return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode="symmetric")

    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    return out_image, (out_mask > 0) * 1

class SimpleLama:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        model_path = hf_hub_download("okaris/simple-lama", "big-lama.pt")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray) -> Image.Image:
        image, mask = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(cur_res)

def main(image_path: str, mask_path: str, out_path: str):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if image.size != mask.size:
        raise ValueError(f"Image size {image.size} does not match mask size {mask.size}")

    width, height = image.size
    image_size_mp = (width * height) / 1_000_000
    
    if image_size_mp > 2:
        print(f"Warning: Input image is {image_size_mp:.2f} MP, which is larger than 2 MP.")
        print("Larger images may not result in better outputs and could increase processing time.")

    lama = SimpleLama()
    result = lama(image, mask)
    result.save(out_path)
    print(f"Inpainted image saved to {out_path}")

def main_cli():
    parser = argparse.ArgumentParser(description="Apply LaMa inpainting using given image and mask.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image (RGB)")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to input mask (Binary 1-CH Image. Pixels with value 255 will be inpainted)")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    args = parser.parse_args()

    main(args.image_path, args.mask_path, args.output)

if __name__ == "__main__":
    main_cli()