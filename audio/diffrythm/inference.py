# Copyright (c) 2025 ASLP-LAB
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir
sys.path.insert(0, str(project_root))

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional
import torch
import torchaudio
from einops import rearrange
import json
from huggingface_hub import hf_hub_download
from muq import MuQMuLan

from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)
from model import DiT, CFM
from g2p.g2p_generation import chn_eng_g2p

class AppInput(BaseAppInput):
    lyrics: str = Field(description="The lyrics to generate music for in LRC format")
    style_prompt: Optional[str] = Field("", description="Text prompt describing the desired music style")
    reference_audio: Optional[File] = Field(None, description="Reference audio file to use as style")
    audio_length: int = Field(95, description="Length of generated song in seconds (95 or 285)")
    edit_mode: bool = Field(False, description="Whether to use edit mode")
    reference_song: Optional[File] = Field(None, description="Reference song for editing")
    edit_segments: Optional[str] = Field(None, description="Time segments to edit in format [[start1,end1],...]")
    batch_size: int = Field(1, description="Number of songs to generate per batch")
    chunked_processing: bool = Field(False, description="Whether to use chunked processing")

class AppOutput(BaseAppOutput):
    generated_audio: File = Field(description="The generated audio file")

class CNENTokenizer:
    def __init__(self):
        with open("./g2p/g2p/vocab.json", "r", encoding='utf-8') as file:
            self.phone2id: dict = json.load(file)["vocab"]
        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x + 1 for x in token]
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x - 1] for x in token])

def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    pred_frames,
    batch_infer_num,
    chunked=False,
):
    with torch.inference_mode():
        latents, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
            latent_pred_segments=pred_frames,
            batch_infer_num=batch_infer_num
        )

        outputs = []
        for latent in latents:
            latent = latent.to(torch.float32)
            latent = latent.transpose(1, 2)  # [b d t]

            output = decode_audio(latent, vae_model, chunked=chunked)

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            # Peak normalize, clip, convert to int16, and save to file
            output = (
                output.to(torch.float32)
                .div(torch.max(torch.abs(output)))
                .clamp(-1, 1)
                .mul(32767)
                .to(torch.int16)
                .cpu()
            )
            outputs.append(output)

        return outputs

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize models and resources."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models
        self.max_frames = 2048  # Default for 95s audio
        self.cfm, self.tokenizer, self.muq, self.vae = prepare_model(
            self.max_frames, 
            self.device
        )

        # Constants
        self.sampling_rate = 44100
        self.downsample_rate = 2048
        self.io_channels = 2

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate music from the input data."""
        # Update max_frames based on audio length
        if input_data.audio_length == 285:
            self.max_frames = 6144

        # Process lyrics
        lrc_tokens, start_time = get_lrc_token(
            self.max_frames,
            input_data.lyrics,
            self.tokenizer,
            self.device
        )

        # Get style prompt
        style_prompt = get_style_prompt(
            self.muq,
            wav_path=input_data.reference_audio.path if input_data.reference_audio else None,
            prompt=input_data.style_prompt if input_data.style_prompt else None
        )
        negative_prompt = get_negative_style_prompt(self.device)

        # Get reference latent for editing if needed
        latent_prompt, pred_frames = get_reference_latent(
            self.device,
            self.max_frames,
            input_data.edit_mode,
            input_data.edit_segments,
            input_data.reference_song.path if input_data.reference_song else None,
            self.vae
        )

        # Generate audio using the inference function
        s_t = time.time()
        generated_songs = inference(
            cfm_model=self.cfm,
            vae_model=self.vae,
            cond=latent_prompt,
            text=lrc_tokens,
            duration=self.max_frames,
            style_prompt=style_prompt,
            negative_style_prompt=negative_prompt,
            start_time=start_time,
            pred_frames=pred_frames,
            chunked=input_data.chunked_processing,
            batch_infer_num=input_data.batch_size
        )
        e_t = time.time() - s_t
        logger.info(f"Inference cost {e_t:.2f} seconds")

        # Save output
        output_path = "/tmp/generated_audio.wav"
        torchaudio.save(
            output_path,
            generated_songs[0],  # Save first generated sample
            sample_rate=self.sampling_rate
        )

        return AppOutput(generated_audio=File(path=output_path))
