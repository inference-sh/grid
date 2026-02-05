import numpy as np
import torch
import cv2
from PIL import Image

import comfy.model_management as model_management
import comfy.utils

# ----------------------------
#  Custom Image Postprocess Node
# ----------------------------
class ImagePostprocessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_std": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.001}),
                "clahe_clip": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "tile": ("INT", {"default": 8, "min": 1, "max": 64}),
                "cutoff": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "fstrength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phase_perturb": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.001}),
                "randomness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "perturb": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 0.05, "step": 0.001}),
                "fft_mode": (["auto", "ref", "model"],),
                "fft_alpha": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "radial_smooth": ("INT", {"default": 5, "min": 0, "max": 50}),
                "jpeg_cycles": ("INT", {"default": 1, "min": 0, "max": 10}),
                "jpeg_qmin": ("INT", {"default": 88, "min": 1, "max": 100}),
                "jpeg_qmax": ("INT", {"default": 96, "min": 1, "max": 100}),
                "vignette_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chroma_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "iso_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "read_noise": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "hot_pixel_prob": ("FLOAT", {"default": 1e-6, "min": 0.0, "max": 1.0, "step": 1e-6}),
                "banding_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "motion_blur_kernel": ("INT", {"default": 1, "min": 1, "max": 51}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1}),
                "sim_camera": ("BOOL", {"default": False}),
                "no_no_bayer": ("BOOL", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/postprocess"

    def process(self, image, **kwargs):
        # convert from torch to numpy
        img = (255.0 * image[0].cpu().numpy()).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Example: add Gaussian noise
        if kwargs["noise_std"] > 0:
            noise = np.random.normal(0, kwargs["noise_std"]*255, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Example: CLAHE
        if kwargs["clahe_clip"] > 0:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=kwargs["clahe_clip"], tileGridSize=(kwargs["tile"], kwargs["tile"]))
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # TODO: implement Fourier filtering, camera sim, vignette, etc.
        #       using your listed parameters.

        # Convert back to torch tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        return (out,)


# ----------------------------
#  Register node
# ----------------------------
NODE_CLASS_MAPPINGS = {
    "ImagePostprocessNode": ImagePostprocessNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePostprocessNode": "Image Postprocess"
}
