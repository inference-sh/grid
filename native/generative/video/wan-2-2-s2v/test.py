import numpy as np
import torch, os
from diffusers import AutoencoderKLWan, WanSpeechToVideoPipeline
from diffusers.utils import export_to_video, load_image, load_audio, load_video
from transformers import Wav2Vec2ForCTC

model_id = "Wan-AI/Wan2.2-S2V-14B-Diffusers"  # will be official
model_id = "tolgacangoz/Wan2.2-S2V-14B-Diffusers"
audio_encoder = Wav2Vec2ForCTC.from_pretrained(model_id, subfolder="audio_encoder", dtype=torch.float32).to("cuda")
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to("cuda")
pipe = WanSpeechToVideoPipeline.from_pretrained(
    model_id, vae=vae, audio_encoder=audio_encoder, torch_dtype=torch.bfloat16,
)#.to("cuda")
pipe.enable_model_cpu_offload()
pipe.transformer.set_attention_backend("flash")

first_frame = load_image("https://raw.githubusercontent.com/Wan-Video/Wan2.2/refs/heads/main/examples/i2v_input.JPG")
audio, sampling_rate = load_audio("https://github.com/Wan-Video/Wan2.2/raw/refs/heads/main/examples/talk.wav")

import math

def get_size_less_than_area(height,
                            width,
                            target_area=1024 * 704,
                            divisor=64):
    if height * width <= target_area:
        # If the original image area is already less than or equal to the target,
        # no resizing is needed—just padding. Still need to ensure that the padded area doesn't exceed the target.
        max_upper_area = target_area
        min_scale = 0.1
        max_scale = 1.0
    else:
        # Resize to fit within the target area and then pad to multiples of `divisor`
        max_upper_area = target_area  # Maximum allowed total pixel count after padding
        d = divisor - 1
        b = d * (height + width)
        a = height * width
        c = d**2 - max_upper_area

        # Calculate scale boundaries using quadratic equation
        min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (
            2 * a)  # Scale when maximum padding is applied
        max_scale = math.sqrt(max_upper_area /
                                (height * width))  # Scale without any padding

    # We want to choose the largest possible scale such that the final padded area does not exceed max_upper_area
    # Use binary search-like iteration to find this scale
    find_it = False
    for i in range(100):
        scale = max_scale - (max_scale - min_scale) * i / 100
        new_height, new_width = int(height * scale), int(width * scale)

        # Pad to make dimensions divisible by 64
        pad_height = (64 - new_height % 64) % 64
        pad_width = (64 - new_width % 64) % 64
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_height, padded_width = new_height + pad_height, new_width + pad_width

        if padded_height * padded_width <= max_upper_area:
            find_it = True
            break

    if find_it:
        return padded_height, padded_width
    else:
        # Fallback: calculate target dimensions based on aspect ratio and divisor alignment
        aspect_ratio = width / height
        target_width = int(
            (target_area * aspect_ratio)**0.5 // divisor * divisor)
        target_height = int(
            (target_area / aspect_ratio)**0.5 // divisor * divisor)

        # Ensure the result is not larger than the original resolution
        if target_width >= width or target_height >= height:
            target_width = int(width // divisor * divisor)
            target_height = int(height // divisor * divisor)

        return target_height, target_width

def aspect_ratio_resize(image, pipe, max_area=720 * 1280):
    height, width = get_size_less_than_area(image.size[1], image.size[0])
    image = image.resize((width, height))
    return image, height, width

first_frame, height, width = aspect_ratio_resize(first_frame, pipe)

prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."

output = pipe(
    image=first_frame, audio=audio, sampling_rate=sampling_rate,
    prompt=prompt, height=height, width=width, num_frames_per_chunk=81,
).frames[0]
export_to_video(output, "video.mp4", fps=16)

import logging, shutil, subprocess

def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # set logging
    logging.basicConfig(level=logging.INFO)

    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        logging.info("Start merging video and audio...")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        logging.info(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        logging.error(f"merge_video_audio failed with error: {e}")

import requests, tempfile
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT

response = requests.get(audio, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT)
with tempfile.NamedTemporaryFile(delete=False) as talk:
    for chunk in response.iter_content(chunk_size=8192):
        talk.write(chunk)
    talk_file = talk.name

merge_video_audio("video.mp4", talk_file)

