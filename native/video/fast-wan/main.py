import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel
import gradio as gr
import tempfile
import spaces
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import random

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
LORA_REPO_ID = "Kijai/WanVideo_comfy"
LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"

image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.to("cuda")

causvid_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)
pipe.load_lora_weights(causvid_path, adapter_name="causvid_lora")
pipe.set_adapters(["causvid_lora"], adapter_weights=[0.95])
pipe.fuse_lora()

MOD_VALUE = 32
DEFAULT_H_SLIDER_VALUE = 512
DEFAULT_W_SLIDER_VALUE = 896
NEW_FORMULA_MAX_AREA = 480.0 * 832.0 

SLIDER_MIN_H, SLIDER_MAX_H = 128, 896
SLIDER_MIN_W, SLIDER_MAX_W = 128, 896
MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 24
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 81 

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards, watermark, text, signature"


def _calculate_new_dimensions_wan(pil_image, mod_val, calculation_max_area,
                                 min_slider_h, max_slider_h,
                                 min_slider_w, max_slider_w,
                                 default_h, default_w):
    orig_w, orig_h = pil_image.size
    if orig_w <= 0 or orig_h <= 0:
        return default_h, default_w

    aspect_ratio = orig_h / orig_w
    
    calc_h = round(np.sqrt(calculation_max_area * aspect_ratio))
    calc_w = round(np.sqrt(calculation_max_area / aspect_ratio))

    calc_h = max(mod_val, (calc_h // mod_val) * mod_val)
    calc_w = max(mod_val, (calc_w // mod_val) * mod_val)
    
    new_h = int(np.clip(calc_h, min_slider_h, (max_slider_h // mod_val) * mod_val))
    new_w = int(np.clip(calc_w, min_slider_w, (max_slider_w // mod_val) * mod_val))
    
    return new_h, new_w

def handle_image_upload_for_dims_wan(uploaded_pil_image, current_h_val, current_w_val):
    if uploaded_pil_image is None:
        return gr.update(value=DEFAULT_H_SLIDER_VALUE), gr.update(value=DEFAULT_W_SLIDER_VALUE)
    try:
        new_h, new_w = _calculate_new_dimensions_wan(
            uploaded_pil_image, MOD_VALUE, NEW_FORMULA_MAX_AREA,
            SLIDER_MIN_H, SLIDER_MAX_H, SLIDER_MIN_W, SLIDER_MAX_W,
            DEFAULT_H_SLIDER_VALUE, DEFAULT_W_SLIDER_VALUE
        )
        return gr.update(value=new_h), gr.update(value=new_w)
    except Exception as e:
        gr.Warning("Error attempting to calculate new dimensions")
        return gr.update(value=DEFAULT_H_SLIDER_VALUE), gr.update(value=DEFAULT_W_SLIDER_VALUE)

def get_duration(input_image, prompt, height, width, 
                   negative_prompt, duration_seconds,
                   guidance_scale, steps,
                   seed, randomize_seed, 
                   progress):
    if steps > 4 and duration_seconds > 2:
        return 90
    elif steps > 4 or duration_seconds > 2:
        return 75
    else:
        return 60

@spaces.GPU(duration=get_duration)
def generate_video(input_image, prompt, height, width, 
                   negative_prompt=default_negative_prompt, duration_seconds = 2,
                   guidance_scale = 1, steps = 4,
                   seed = 42, randomize_seed = False, 
                   progress=gr.Progress(track_tqdm=True)):
    """
    Generate a video from an input image using the Wan 2.1 I2V model with CausVid LoRA.
    
    This function takes an input image and generates a video animation based on the provided
    prompt and parameters. It uses the Wan 2.1 14B Image-to-Video model with CausVid LoRA
    for fast generation in 4-8 steps.
    
    Args:
        input_image (PIL.Image): The input image to animate. Will be resized to target dimensions.
        prompt (str): Text prompt describing the desired animation or motion.
        height (int): Target height for the output video. Will be adjusted to multiple of MOD_VALUE (32).
        width (int): Target width for the output video. Will be adjusted to multiple of MOD_VALUE (32).
        negative_prompt (str, optional): Negative prompt to avoid unwanted elements. 
            Defaults to default_negative_prompt (contains unwanted visual artifacts).
        duration_seconds (float, optional): Duration of the generated video in seconds.
            Defaults to 2. Clamped between MIN_FRAMES_MODEL/FIXED_FPS and MAX_FRAMES_MODEL/FIXED_FPS.
        guidance_scale (float, optional): Controls adherence to the prompt. Higher values = more adherence.
            Defaults to 1.0. Range: 0.0-20.0.
        steps (int, optional): Number of inference steps. More steps = higher quality but slower.
            Defaults to 4. Range: 1-30.
        seed (int, optional): Random seed for reproducible results. Defaults to 42.
            Range: 0 to MAX_SEED (2147483647).
        randomize_seed (bool, optional): Whether to use a random seed instead of the provided seed.
            Defaults to False.
        progress (gr.Progress, optional): Gradio progress tracker. Defaults to gr.Progress(track_tqdm=True).
    
    Returns:
        tuple: A tuple containing:
            - video_path (str): Path to the generated video file (.mp4)
            - current_seed (int): The seed used for generation (useful when randomize_seed=True)
    
    Raises:
        gr.Error: If input_image is None (no image uploaded).
    
    Note:
        - The function automatically resizes the input image to the target dimensions
        - Frame count is calculated as duration_seconds * FIXED_FPS (24)
        - Output dimensions are adjusted to be multiples of MOD_VALUE (32)
        - The function uses GPU acceleration via the @spaces.GPU decorator
        - Generation time varies based on steps and duration (see get_duration function)
    """
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    target_h = max(MOD_VALUE, (int(height) // MOD_VALUE) * MOD_VALUE)
    target_w = max(MOD_VALUE, (int(width) // MOD_VALUE) * MOD_VALUE)
    
    num_frames = np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL)
    
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    resized_image = input_image.resize((target_w, target_h))

    with torch.inference_mode():
        output_frames_list = pipe(
            image=resized_image, prompt=prompt, negative_prompt=negative_prompt,
            height=target_h, width=target_w, num_frames=num_frames,
            guidance_scale=float(guidance_scale), num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed)
        ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)
    return video_path, current_seed

with gr.Blocks() as demo:
    gr.Markdown("# Fast 4 steps Wan 2.1 I2V (14B) with CausVid LoRA")
    gr.Markdown("[CausVid](https://github.com/tianweiy/CausVid) is a distilled version of Wan 2.1 to run faster in just 4-8 steps, [extracted as LoRA by Kijai](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_CausVid_14B_T2V_lora_rank32.safetensors) and is compatible with 🧨 diffusers")
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Input Image (auto-resized to target H/W)")
            prompt_input = gr.Textbox(label="Prompt", value=default_prompt_i2v)
            duration_seconds_input = gr.Slider(minimum=round(MIN_FRAMES_MODEL/FIXED_FPS,1), maximum=round(MAX_FRAMES_MODEL/FIXED_FPS,1), step=0.1, value=2, label="Duration (seconds)", info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps.")
            
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(label="Negative Prompt", value=default_negative_prompt, lines=3)
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42, interactive=True)
                randomize_seed_checkbox = gr.Checkbox(label="Randomize seed", value=True, interactive=True)
                with gr.Row():
                    height_input = gr.Slider(minimum=SLIDER_MIN_H, maximum=SLIDER_MAX_H, step=MOD_VALUE, value=DEFAULT_H_SLIDER_VALUE, label=f"Output Height (multiple of {MOD_VALUE})")
                    width_input = gr.Slider(minimum=SLIDER_MIN_W, maximum=SLIDER_MAX_W, step=MOD_VALUE, value=DEFAULT_W_SLIDER_VALUE, label=f"Output Width (multiple of {MOD_VALUE})")
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=4, label="Inference Steps") 
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=20.0, step=0.5, value=1.0, label="Guidance Scale", visible=False)

            generate_button = gr.Button("Generate Video", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Generated Video", autoplay=True, interactive=False)

    input_image_component.upload(
        fn=handle_image_upload_for_dims_wan,
        inputs=[input_image_component, height_input, width_input],
        outputs=[height_input, width_input]
    )
    
    input_image_component.clear( 
        fn=handle_image_upload_for_dims_wan,
        inputs=[input_image_component, height_input, width_input],
        outputs=[height_input, width_input]
    )
    
    ui_inputs = [
        input_image_component, prompt_input, height_input, width_input,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, steps_slider, seed_input, randomize_seed_checkbox
    ]
    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, seed_input])

    gr.Examples(
        examples=[ 
            ["peng.png", "a penguin playfully dancing in the snow, Antarctica", 896, 512],
            ["forg.jpg", "the frog jumps around", 448, 832],
        ],
        inputs=[input_image_component, prompt_input, height_input, width_input], outputs=[video_output, seed_input], fn=generate_video, cache_examples="lazy"
    )

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True)
