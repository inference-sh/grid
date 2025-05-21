# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

import os
import subprocess
from tqdm import tqdm


def inference_video_from_dir(input_dir, output_dir, unet_config_path, ckpt_path):
    os.makedirs(output_dir, exist_ok=True)
    video_names = sorted([f for f in os.listdir(input_dir) if f.endswith(".mp4")])
    for video_name in tqdm(video_names):
        video_path = os.path.join(input_dir, video_name)
        audio_path = os.path.join(input_dir, video_name.replace(".mp4", "_audio.wav"))
        video_out_path = os.path.join(output_dir, video_name.replace(".mp4", "_out.mp4"))
        inference_command = f"python inference.py --unet_config_path {unet_config_path} --video_path {video_path} --audio_path {audio_path} --video_out_path {video_out_path} --inference_ckpt_path {ckpt_path} --seed 1247"
        subprocess.run(inference_command, shell=True)


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/HDTF/segmented/cross"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/HDTF/segmented/latentsync_cross"
    unet_config_path = "configs/unet/unet_latent_16_diffusion.yaml"
    ckpt_path = "output/unet/train-2024_10_08-16:23:43/checkpoints/checkpoint-1920000.pt"

    inference_video_from_dir(input_dir, output_dir, unet_config_path, ckpt_path)
