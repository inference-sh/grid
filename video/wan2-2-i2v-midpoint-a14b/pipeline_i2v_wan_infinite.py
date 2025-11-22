# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> import numpy as np
        >>> from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image
        >>> from transformers import CLIPVisionModel

        >>> # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        >>> image_encoder = CLIPVisionModel.from_pretrained(
        ...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
        ... )
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanImageToVideoPipeline.from_pretrained(
        ...     model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )
        >>> # You can also use additional initial images for conditioning:
        >>> # initial_images = [load_image("path/to/image1.jpg"), load_image("path/to/image2.jpg")]
        >>> max_area = 480 * 832
        >>> aspect_ratio = image.height / image.width
        >>> mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        >>> height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        >>> width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        >>> image = image.resize((width, height))
        >>> # For initial images: initial_images = [img.resize((width, height)) for img in initial_images]
        >>> prompt = (
        ...     "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        ...     "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        ... )
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     image=image,  # Main conditioning image
        ...     initial_images=initial_images,  # Additional conditioning frames (optional)
        ...     midpoint_conditioning=False,  # Set to True to place image at midpoint
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=height,
        ...     width=width,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class WanImageToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        transformer_2 ([`WanTransformer3DModel`], *optional*):
            Conditional Transformer to denoise the input latents during the low-noise stage. In two-stage denoising,
            `transformer` handles high-noise stages and `transformer_2` handles low-noise stages. If not provided, only
            `transformer` is used.
        boundary_ratio (`float`, *optional*, defaults to `None`):
            Ratio of total timesteps to use as the boundary for switching between transformers in two-stage denoising.
            The actual boundary timestep is calculated as `boundary_ratio * num_train_timesteps`. When provided,
            `transformer` handles timesteps >= boundary_timestep and `transformer_2` handles timesteps <
            boundary_timestep. If `None`, only `transformer` is used for the entire denoising process.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "condition"]
    _optional_components = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer: WanTransformer3DModel = None,
        transformer_2: WanTransformer3DModel = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: Union[PipelineImageInput, List[PipelineImageInput]],
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        # Handle both single image and list of images
        if isinstance(image, list):
            # Process all images together
            processed_images = self.image_processor(images=image, return_tensors="pt").to(device)
        else:
            processed_images = self.image_processor(images=image, return_tensors="pt").to(device)
        
        image_embeds = self.image_encoder(**processed_images, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        initial_images=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        
        if initial_images is not None:
            if not isinstance(initial_images, list):
                raise ValueError(f"`initial_images` has to be a list but is {type(initial_images)}")
            for i, img in enumerate(initial_images):
                if not isinstance(img, torch.Tensor) and not isinstance(img, PIL.Image.Image):
                    raise ValueError(f"`initial_images[{i}]` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(img)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if self.config.boundary_ratio is not None and image_embeds is not None:
            raise ValueError("Cannot forward `image_embeds` when the pipeline's `boundary_ratio` is not configured.")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        initial_images: Optional[List[PipelineImageInput]] = None,
        last_image: Optional[torch.Tensor] = None,
        midpoint_conditioning: bool = False,
        external_condition_latents: Optional[torch.Tensor] = None,
        external_mask_lat_size: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # If external conditioning is provided, use it directly and bypass image conditioning
        if external_condition_latents is not None and external_mask_lat_size is not None:
            latent_condition = external_condition_latents.to(device=device, dtype=dtype)
            
            if self.config.expand_timesteps:
                # For expand_timesteps mode, create appropriate mask
                first_frame_mask = torch.ones(
                    1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
                )
                # The external mask tells us which frames are conditioned
                # Convert from frame-level mask to latent-level mask
                external_mask_latent = external_mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
                external_mask_latent = external_mask_latent.transpose(1, 2)
                # Invert mask: 0 where we have conditioning, 1 where we generate
                first_frame_mask = 1 - external_mask_latent[0:1, 0:1]  # Take first batch/channel
                return latents, latent_condition, first_frame_mask
            else:
                # Use the provided mask directly
                mask_lat_size = external_mask_lat_size.to(device)
                first_frame_mask = mask_lat_size[:, :, 0:1]
                first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
                mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
                mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
                mask_lat_size = mask_lat_size.transpose(1, 2)
                return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

        # Start with the main image
        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]
        
        # Handle midpoint conditioning
        if midpoint_conditioning:
            # Calculate midpoint position
            midpoint_idx = num_frames // 2
            
            # Create video condition with main image at midpoint
            video_condition_frames = []
            
            # Add zeros before midpoint
            if midpoint_idx > 0:
                zero_frames_before = image.new_zeros(
                    image.shape[0], image.shape[1], midpoint_idx, height, width
                )
                video_condition_frames.append(zero_frames_before)
            
            # Add main image at midpoint
            video_condition_frames.append(image)
            
            # Add initial_images after main image (if any)
            if initial_images is not None:
                for init_img in initial_images:
                    init_img_tensor = init_img.unsqueeze(2)
                    video_condition_frames.append(init_img_tensor)
            
            # Calculate remaining frames needed
            frames_used = midpoint_idx + 1 + (len(initial_images) if initial_images else 0)
            
            # Add last_image if provided and there's space
            if last_image is not None and frames_used < num_frames:
                last_image = last_image.unsqueeze(2)
                video_condition_frames.append(last_image)
                frames_used += 1
            
            # Fill remaining frames with zeros
            remaining_frames = num_frames - frames_used
            if remaining_frames > 0:
                zero_frames_after = image.new_zeros(
                    image.shape[0], image.shape[1], remaining_frames, height, width
                )
                video_condition_frames.append(zero_frames_after)
            
            video_condition = torch.cat(video_condition_frames, dim=2)
            num_conditioning_frames = 1 + (len(initial_images) if initial_images else 0) + (1 if last_image is not None and frames_used <= num_frames else 0)
            
        else:
            # Original logic: main image at beginning
            # Collect all initial frames (main image + initial_images)
            all_initial_frames = [image]
            
            if initial_images is not None:
                # Add each initial image to the sequence
                for init_img in initial_images:
                    init_img_tensor = init_img.unsqueeze(2)  # [batch_size, channels, 1, height, width]
                    all_initial_frames.append(init_img_tensor)
            
            # Concatenate all initial frames
            if len(all_initial_frames) > 1:
                initial_frames = torch.cat(all_initial_frames, dim=2)  # [batch_size, channels, N, height, width]
                num_conditioning_frames = len(all_initial_frames)
                
                if num_conditioning_frames > num_frames:
                    raise ValueError(f"Number of initial images ({num_conditioning_frames}) cannot exceed num_frames ({num_frames})")
            else:
                initial_frames = image
                num_conditioning_frames = 1

            # Create video condition based on configuration
            if self.config.expand_timesteps:
                video_condition = initial_frames
            elif last_image is None:
                # Fill remaining frames with zeros
                remaining_frames = num_frames - num_conditioning_frames
                if remaining_frames > 0:
                    zero_frames = image.new_zeros(
                        image.shape[0], image.shape[1], remaining_frames, height, width
                    )
                    video_condition = torch.cat([initial_frames, zero_frames], dim=2)
                else:
                    video_condition = initial_frames
            else:
                # Add last_image at the end
                last_image = last_image.unsqueeze(2)
                remaining_frames = num_frames - num_conditioning_frames - 1
                if remaining_frames > 0:
                    zero_frames = image.new_zeros(
                        image.shape[0], image.shape[1], remaining_frames, height, width
                    )
                    video_condition = torch.cat([initial_frames, zero_frames, last_image], dim=2)
                else:
                    video_condition = torch.cat([initial_frames, last_image], dim=2)
                num_conditioning_frames += 1
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        if self.config.expand_timesteps:
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
            )
            
            if midpoint_conditioning:
                # For midpoint conditioning, mask based on conditioning frame positions
                midpoint_latent_idx = min((num_frames // 2) // self.vae_scale_factor_temporal, num_latent_frames - 1)
                # Mask out the conditioning frames
                first_frame_mask[:, :, midpoint_latent_idx] = 0
                # If there are initial_images, mask those too
                if initial_images is not None:
                    for i in range(len(initial_images)):
                        conditioning_latent_idx = min((midpoint_latent_idx + 1 + i), num_latent_frames - 1)
                        first_frame_mask[:, :, conditioning_latent_idx] = 0
            else:
                # Original logic: mask out the initial frames based on number of conditioning frames
                num_conditioning_latent_frames = min((num_conditioning_frames - 1) // self.vae_scale_factor_temporal + 1, num_latent_frames)
                first_frame_mask[:, :, :num_conditioning_latent_frames] = 0
            
            return latents, latent_condition, first_frame_mask

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if midpoint_conditioning:
            # For midpoint conditioning, mask the conditioning frames
            midpoint_idx = num_frames // 2
            # Mask out the main image at midpoint
            mask_lat_size[:, :, midpoint_idx] = 0
            
            # Mask out initial_images if present
            if initial_images is not None:
                for i in range(len(initial_images)):
                    conditioning_idx = midpoint_idx + 1 + i
                    if conditioning_idx < num_frames:
                        mask_lat_size[:, :, conditioning_idx] = 0
            
            # Mask out last_image if present
            if last_image is not None:
                frames_used = midpoint_idx + 1 + (len(initial_images) if initial_images else 0)
                if frames_used < num_frames:
                    mask_lat_size[:, :, frames_used] = 0
        else:
            # Original logic: create mask based on conditioning frames and last image
            if last_image is None:
                # Mask out frames after the conditioning frames
                if num_conditioning_frames < num_frames:
                    mask_lat_size[:, :, list(range(num_conditioning_frames, num_frames))] = 0
            else:
                # Mask out frames between conditioning frames and last image
                if num_conditioning_frames < num_frames - 1:
                    mask_lat_size[:, :, list(range(num_conditioning_frames, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        initial_images: Optional[List[PipelineImageInput]] = None,
        last_image: Optional[torch.Tensor] = None,
        midpoint_conditioning: bool = False,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        dual_continuous_generation: bool = False,
        overlap_frames: int = 16,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image or a `torch.Tensor`. This will be used as the first frame of the video.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            initial_images (`List[PipelineImageInput]`, *optional*):
                A list of additional images to use as conditioning frames after the first image. These will be placed as frames 2, 3, 4, etc. in the video conditioning sequence.
            midpoint_conditioning (`bool`, *optional*, defaults to `False`):
                If True, the main image will be placed at the midpoint of the video conditioning sequence instead of at the beginning. This allows the model to generate frames both before and after the conditioning image.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.
            dual_continuous_generation (`bool`, *optional*, defaults to `False`):
                Enable dual continuous video generation with overlapping sequences.
            overlap_frames (`int`, *optional*, defaults to 16):
                Number of overlapping frames between dual sequences in continuous generation.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            initial_images,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # Check if dual continuous generation is requested
        if dual_continuous_generation:
            return self.dual_continuous_generation(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                overlap_frames=overlap_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image_embeds=image_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
            )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # only wan 2.1 i2v transformer accepts image_embeds
        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                # Create list of all images to encode
                images_to_encode = [image]
                if initial_images is not None:
                    images_to_encode.extend(initial_images)
                if last_image is not None:
                    images_to_encode.append(last_image)
                
                # Encode all images together
                image_embeds = self.encode_image(images_to_encode, device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        
        # Preprocess the main image
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        
        # Preprocess initial_images if provided
        if initial_images is not None:
            processed_initial_images = []
            for init_img in initial_images:
                processed_img = self.video_processor.preprocess(init_img, height=height, width=width).to(device, dtype=torch.float32)
                processed_initial_images.append(processed_img)
            initial_images = processed_initial_images
        
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            initial_images,
            last_image,
            midpoint_conditioning,
            None,  # external_condition_latents - not used in normal mode
            None,  # external_mask_lat_size - not used in normal mode
        )
        if self.config.expand_timesteps:
            # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                if self.config.expand_timesteps:
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if self.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

    def dual_continuous_generation(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        overlap_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
    ):
        """
        Generate two continuous video sequences with overlapping conditioning.
        
        Returns a video roughly 2x the length of num_frames with smooth transitions.
        """
        device = self._execution_device
        
        # Preprocess image
        processed_image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        
        # Prepare initial latents for both sequences
        num_channels_latents = self.vae.config.z_dim
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        
        shape = (1, num_channels_latents, num_latent_frames, latent_height, latent_width)
        
        # Initialize latents for both sequences
        latents_A = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        latents_B = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        
        # Prepare text embeddings
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance_scale > 1,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        
        # Prepare conditioning using the standard pipeline method
        # For sequence A: condition with the input image at the beginning
        latents_A_output = self.prepare_latents(
            processed_image, 1, num_channels_latents, height, width, num_frames,
            torch.float32, device, generator, None, None, None, False, None, None
        )
        if self.config.expand_timesteps:
            _, condition_A, mask_A = latents_A_output
        else:
            _, condition_with_mask_A = latents_A_output
            # Split condition and mask - mask has vae_scale_factor_temporal channels
            mask_channels = self.vae_scale_factor_temporal
            mask_A = condition_with_mask_A[:, :mask_channels]  # First vae_scale_factor_temporal channels are mask
            condition_A = condition_with_mask_A[:, mask_channels:]  # Remaining channels are condition
        
        # For sequence B: start with no conditioning (will be updated from A)
        if self.config.expand_timesteps:
            condition_B = torch.zeros_like(condition_A)
            mask_B = torch.ones_like(mask_A)
        else:
            condition_B = torch.zeros_like(condition_A)
            mask_B = torch.ones_like(mask_A)
        
        # Scheduler setup
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
        
        # Denoising loop with cross-conditioning
        for i, t in enumerate(timesteps):
            # Step sequence A
            latents_A = self._denoise_step(
                latents_A, condition_A, mask_A, t, prompt_embeds, negative_prompt_embeds,
                image_embeds, guidance_scale, attention_kwargs
            )
            
            # Step sequence B  
            latents_B = self._denoise_step(
                latents_B, condition_B, mask_B, t, prompt_embeds, negative_prompt_embeds,
                image_embeds, guidance_scale, attention_kwargs
            )
            
            # Update cross-conditioning after each step
            if i < len(timesteps) - 1:  # Don't update on last step
                # Update B's conditioning from A's tail
                condition_B, mask_B = self._update_conditioning_from_sequence(
                    latents_A, condition_B, mask_B, overlap_frames, latents_mean, latents_std,
                    num_frames, height, width, position="tail"
                )
                
                # Update A's conditioning from B's head  
                condition_A, mask_A = self._update_conditioning_from_sequence(
                    latents_B, condition_A, mask_A, overlap_frames, latents_mean, latents_std,
                    num_frames, height, width, position="head"
                )
        
        # Decode both sequences
        if output_type != "latent":
            latents_A = latents_A.to(self.vae.dtype)
            latents_B = latents_B.to(self.vae.dtype)
            
            latents_A = latents_A / latents_std + latents_mean
            latents_B = latents_B / latents_std + latents_mean
            
            video_A = self.vae.decode(latents_A, return_dict=False)[0]
            video_B = self.vae.decode(latents_B, return_dict=False)[0]
            
            video_A = self.video_processor.postprocess_video(video_A, output_type=output_type)
            video_B = self.video_processor.postprocess_video(video_B, output_type=output_type)
            
            # Concatenate sequences with overlap blending
            final_video = self._blend_sequences(video_A, video_B, overlap_frames)
        else:
            final_video = [latents_A, latents_B]
        
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (final_video,)
        
        return WanPipelineOutput(frames=final_video)

    def _denoise_step(self, latents, condition, mask, timestep, prompt_embeds, negative_prompt_embeds, image_embeds, guidance_scale, attention_kwargs):
        """Single denoising step for one sequence."""
        if self.config.expand_timesteps:
            # For expand_timesteps mode, blend latents and condition with mask
            latent_model_input = (1 - mask) * condition + mask * latents
        else:
            # For normal mode, concatenate mask and condition, then concatenate with latents
            condition_with_mask = torch.cat([mask, condition], dim=1)
            latent_model_input = torch.cat([latents, condition_with_mask], dim=1)
        
        latent_model_input = latent_model_input.to(prompt_embeds.dtype)
        
        if self.config.expand_timesteps:
            # For expand_timesteps, create per-pixel timesteps
            temp_ts = (mask[0][0][:, ::2, ::2] * timestep).flatten()
            timestep_input = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            timestep_input = timestep.expand(latents.shape[0])
        
        # Conditional prediction
        with self.transformer.cache_context("cond"):
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
        
        # Classifier-free guidance
        if guidance_scale > 1:
            with self.transformer.cache_context("uncond"):
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_input,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        
        # Update latents
        return self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    def _update_conditioning_from_sequence(self, source_latents, target_condition, target_mask, overlap_frames, latents_mean, latents_std, num_frames, height, width, position="tail"):
        """Update conditioning for one sequence from another sequence's latents."""
        # Convert latents to condition space (normalize to VAE space)
        source_condition = (source_latents - latents_mean) * latents_std
        
        # Calculate overlap in latent temporal dimension
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        overlap_latent_frames = min(overlap_frames // self.vae_scale_factor_temporal, num_latent_frames // 2)
        
        if self.config.expand_timesteps:
            # For expand_timesteps mode, work directly with latent frames
            if position == "tail":
                # Take last overlap_latent_frames from source for target's beginning
                source_frames = source_condition[:, :, -overlap_latent_frames:]
                target_condition[:, :, :overlap_latent_frames] = source_frames
                target_mask[:, :, :overlap_latent_frames] = 0  # Mark as conditioned
            else:  # position == "head"
                # Take first overlap_latent_frames from source for target's end
                source_frames = source_condition[:, :, :overlap_latent_frames]
                target_condition[:, :, -overlap_latent_frames:] = source_frames
                target_mask[:, :, -overlap_latent_frames:] = 0  # Mark as conditioned
        else:
            # For normal mode, we need to update the frame-level conditioning
            # Convert latent frames back to frame-level for mask updates
            overlap_frame_indices = list(range(overlap_frames)) if position == "tail" else list(range(num_frames - overlap_frames, num_frames))
            
            if position == "tail":
                # Take last overlap frames from source (in latent space)
                source_frames = source_condition[:, :, -overlap_latent_frames:]
                target_condition[:, :, :overlap_latent_frames] = source_frames
                # Update mask for corresponding frames
                for i in range(min(overlap_frames, num_frames)):
                    target_mask[:, :, i] = 0
            else:  # position == "head"
                # Take first overlap frames from source for target's end
                source_frames = source_condition[:, :, :overlap_latent_frames]
                target_condition[:, :, -overlap_latent_frames:] = source_frames
                # Update mask for corresponding frames
                for i in range(max(0, num_frames - overlap_frames), num_frames):
                    target_mask[:, :, i] = 0
        
        return target_condition, target_mask

    def _blend_sequences(self, video_A, video_B, overlap_frames):
        """Blend two video sequences with smooth overlap transition."""
        # Simple concatenation with overlap blending
        total_frames = len(video_A) + len(video_B) - overlap_frames
        blended_video = []
        
        # Add sequence A
        blended_video.extend(video_A[:-overlap_frames])
        
        # Blend overlap region
        for i in range(overlap_frames):
            alpha = i / (overlap_frames - 1) if overlap_frames > 1 else 0.5
            frame_A = video_A[-(overlap_frames - i)]
            frame_B = video_B[i]
            blended_frame = (1 - alpha) * frame_A + alpha * frame_B
            blended_video.append(blended_frame)
        
        # Add remaining sequence B
        blended_video.extend(video_B[overlap_frames:])
        
        return blended_video

