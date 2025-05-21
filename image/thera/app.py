import pickle
import json
import os

import gradio as gr
from PIL import Image
import numpy as np
import jax

from gradio_dualvision import DualVisionApp
from gradio_dualvision.gradio_patches.radio import Radio
from huggingface_hub import hf_hub_download
from model import build_thera
from super_resolve import process

REPO_ID_EDSR = "prs-eth/thera-edsr-pro"
REPO_ID_RDN = "prs-eth/thera-rdn-pro"
MAX_SIZE = int(os.getenv('THERA_DEMO_CROP', 10_000))

print(f"JAX devices: {jax.devices()}")
print(f"JAX device type: {jax.devices()[0].device_kind}")

model_path = hf_hub_download(repo_id=REPO_ID_EDSR, filename="model.pkl")
with open(model_path, 'rb') as fh:
    check = pickle.load(fh)
    params_edsr, backbone, size = check['model'], check['backbone'], check['size']
    model_edsr = build_thera(3, backbone, size)

model_path = hf_hub_download(repo_id=REPO_ID_RDN, filename="model.pkl")
with open(model_path, 'rb') as fh:
    check = pickle.load(fh)
    params_rdn, backbone, size = check['model'], check['backbone'], check['size']
    model_rdn = build_thera(3, backbone, size)


class TheraApp(DualVisionApp):
    DEFAULT_SCALE = 3.92
    DEFAULT_DO_ENSEMBLE = False
    DEFAULT_MODEL = 'edsr'

    def make_header(self):
        gr.Markdown(
            """
            ## Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields
            <p align="center">
            <a title="Website" href="https://therasr.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/badge/%E2%99%A5%20Project%20-Website-blue">
            </a>
            <a title="arXiv" href="https://arxiv.org/pdf/2311.17643" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/badge/%F0%9F%93%84%20Read%20-Paper-AF3436">
            </a>
            <a title="Github" href="https://github.com/prs-eth/thera" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/github/stars/prs-eth/thera?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
            </a>
            </p>                    
            <p align="center" style="margin-top: 0px;">
                <strong>Upload a photo or select an example below to do arbitrary-scale super-resolution in real time!</strong>
            </p>
            <p align="center" style="margin-top: 0px;">
                <strong style="color: red;">Note: The model has not been trained on input images with JPEG artifacts, so this does not work well.</strong>
            </p>
            <p align="center" style="margin-top: 0px;">
                <strong>Also note: Due to limited viewport size in the browser, the effect is best visible for smaller inputs (e.g. 150x150 px).<br>For larger inputs, it makes sense to zoom in or download the result and compare locally. We're working on a better solution for visualization.</strong>
            </p>
        """
        )

    def build_user_components(self):
        with gr.Row():
            scale = gr.Slider(
                label="Scaling factor",
                minimum=1,
                maximum=6,
                step=0.01,
                value=self.DEFAULT_SCALE,
            )
            model = gr.Radio(
                [
                    ("EDSR", 'edsr'),
                    ("RDN", 'rdn'),
                ],
                label="Backbone",
                value=self.DEFAULT_MODEL,
            )
            do_ensemble = gr.Radio(
                [
                    ("No", False),
                    ("Yes", True),
                ],
                label="Do Ensemble",
                value=self.DEFAULT_DO_ENSEMBLE,
            )
        return {
            "scale": scale,
            "model": model,
            "do_ensemble": do_ensemble,
        }

    def process(self, image_in: Image.Image, **kwargs):
        scale = kwargs.get("scale", self.DEFAULT_SCALE)
        do_ensemble = kwargs.get("do_ensemble", self.DEFAULT_DO_ENSEMBLE)
        model = kwargs.get("model", self.DEFAULT_MODEL)

        if max(*image_in.size) > MAX_SIZE:
            gr.Warning(f"The image has been cropped for better visibility, and to enable a smooth experience for all users.")
            width, height = image_in.size
            crop_width = min(width, MAX_SIZE)
            crop_height = min(height, MAX_SIZE)
            left = (width - crop_width) / 2
            top = (height - crop_height) / 2
            right = left + crop_width
            bottom = top + crop_height
            image_in = image_in.crop((left, top, right, bottom))

        source = np.asarray(image_in) / 255.

        # determine target shape
        target_shape = (
            round(source.shape[0] * scale),
            round(source.shape[1] * scale),
        )

        if model == 'edsr':
            m, p = model_edsr, params_edsr
        elif model == 'rdn':
            m, p = model_rdn, params_rdn
        else:
            raise NotImplementedError('model:', model)

        out = process(source, m, p, target_shape, do_ensemble=do_ensemble)
        out = Image.fromarray(np.asarray(out))

        nearest = image_in.resize(out.size, Image.NEAREST)

        out_modalities = {
            "nearest": nearest,
            "out": out,
        }
        out_settings = {
            'scale': scale,
            'model': model,
            'do_ensemble': do_ensemble,
        }
        return out_modalities, out_settings

    def process_components(
        self, image_in, modality_selector_left, modality_selector_right, **kwargs
    ):
        if image_in is None:
            raise gr.Error("Input image is required")

        image_settings = {}
        if isinstance(image_in, str):
            image_settings_path = image_in + ".settings.json"
            if os.path.isfile(image_settings_path):
                with open(image_settings_path, "r") as f:
                    image_settings = json.load(f)
            image_in = Image.open(image_in).convert("RGB")
        else:
            if not isinstance(image_in, Image.Image):
                raise gr.Error(f"Input must be a PIL image, got {type(image_in)}")
            image_in = image_in.convert("RGB")
        image_settings.update(kwargs)

        results_dict, results_settings = self.process(image_in, **image_settings)

        if not isinstance(results_dict, dict):
            raise gr.Error(
                f"`process` must return a dict[str, PIL.Image]. Got type: {type(results_dict)}"
            )
        if len(results_dict) == 0:
            raise gr.Error("`process` did not return any modalities")
        for k, v in results_dict.items():
            if not isinstance(k, str):
                raise gr.Error(
                    f"Output dict must have string keys. Found key of type {type(k)}: {repr(k)}"
                )
            if k == self.key_original_image:
                raise gr.Error(
                    f"Output dict must not have an '{self.key_original_image}' key; it is reserved for the input"
                )
            if not isinstance(v, Image.Image):
                raise gr.Error(
                    f"Value for key '{k}' must be a PIL Image, got type {type(v)}"
                )
        if len(results_settings) != len(self.input_keys):
            raise gr.Error(
                f"Expected number of settings ({len(self.input_keys)}), returned ({len(results_settings)})"
            )
        if any(k not in results_settings for k in self.input_keys):
            raise gr.Error(f"Mismatching setgings keys")

        results_settings = {
            k: cls(**ctor_args, value=results_settings[k])
            for k, cls, ctor_args in zip(
                self.input_keys, self.input_cls, self.input_kwargs
            )
        }

        results_dict = {
            **results_dict,
            self.key_original_image: image_in,
        }

        results_state = [[v, k] for k, v in results_dict.items()]
        modalities = list(results_dict.keys())

        modality_left = (
            modality_selector_left
            if modality_selector_left in modalities
            else modalities[0]
        )
        modality_right = (
            modality_selector_right
            if modality_selector_right in modalities
            else modalities[1]
        )

        return [
            results_state,  # goes to a gr.Gallery
            [
                results_dict[modality_left],
                results_dict[modality_right],
            ],  # ImageSliderPlus
            Radio(
                choices=modalities,
                value=modality_left,
                label="Left",
                key="Left",
            ),
            Radio(
                choices=modalities if self.left_selector_visible else modalities[1:],
                value=modality_right,
                label="Right",
                key="Right",
            ),
            *results_settings.values(),
        ]


with TheraApp(
        title="Thera Arbitrary-Scale Super-Resolution",
        examples_path="files",
        examples_per_page=12,
        squeeze_canvas=True,
        advanced_settings_can_be_half_width=False,
        #spaces_zero_gpu_enabled=True,
) as demo:
    demo.queue(
        api_open=False,
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
