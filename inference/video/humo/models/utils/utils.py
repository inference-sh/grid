# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp
import json
from omegaconf import OmegaConf

import imageio
import torch
import torchvision
from moviepy.editor import AudioFileClip, VideoClip

__all__ = ['tensor_to_video', 'prepare_json_dataset']
    

def tensor_to_video(tensor, output_video_path, input_audio_path, fps=25):
    """
    Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track from the specified audio file.

    Args:
        tensor (numpy): The Tensor to be converted, shaped [f, h, w, c].
        output_video_path (str): The file path where the output video will be saved.
        input_audio_path (str): The path to the audio file (WAV file) that contains the audio track to be added.
        fps (int): The frame rate of the output video. Default is 30 fps.
    """
    def make_frame(t):
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        return tensor[frame_index]

    video_duration = tensor.shape[0] / fps
    audio_clip = AudioFileClip(input_audio_path)
    audio_duration = audio_clip.duration
    final_duration = min(video_duration, audio_duration)
    audio_clip = audio_clip.subclip(0, final_duration)
    new_video_clip = VideoClip(make_frame, duration=final_duration)
    new_video_clip = new_video_clip.set_audio(audio_clip)
    new_video_clip.write_videofile(output_video_path, fps=fps, audio_codec="aac")


def prepare_json_dataset(json_path):
    samples = []
    with open(json_path, "rb") as f:
        data = json.load(f)
    for itemname, row in data.items():
        text = row['prompt'].strip().replace("_", " ").strip('"')
        audio_path = row['audio_path']
        ref_img_path = [x for x in row['img_paths']]

        samples.append({
            "text": text,
            "ref_img": ref_img_path,
            "audio": audio_path,
            "itemname": itemname
        })
    samples = OmegaConf.create(samples)
    
    return samples