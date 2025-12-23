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
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import cv2
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..utils.audio import melspectrogram
from decord import AudioReader, VideoReader, cpu


class UNetDataset(Dataset):
    def __init__(self, train_data_dir: str, config):
        if config.data.train_fileslist != "":
            with open(config.data.train_fileslist) as file:
                self.video_paths = [line.rstrip() for line in file]
        elif train_data_dir != "":
            self.video_paths = []
            for file in os.listdir(train_data_dir):
                if file.endswith(".mp4"):
                    self.video_paths.append(os.path.join(train_data_dir, file))
        else:
            raise ValueError("data_dir and fileslist cannot be both empty")

        self.resolution = config.data.resolution
        self.num_frames = config.data.num_frames

        if self.num_frames == 16:
            self.mel_window_length = 52
        elif self.num_frames == 5:
            self.mel_window_length = 16
        else:
            raise NotImplementedError("Only support 16 and 5 frames now")

        self.audio_sample_rate = config.data.audio_sample_rate
        self.video_fps = config.data.video_fps
        self.mask = config.data.mask
        self.mask_image = load_fixed_mask(self.resolution)
        self.load_audio_data = config.model.add_audio_layer and config.run.use_syncnet
        self.audio_mel_cache_dir = config.data.audio_mel_cache_dir
        os.makedirs(self.audio_mel_cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.video_paths)

    def read_audio(self, video_path: str):
        ar = AudioReader(video_path, ctx=cpu(self.worker_id), sample_rate=self.audio_sample_rate)
        original_mel = melspectrogram(ar[:].asnumpy().squeeze(0))
        return torch.from_numpy(original_mel)

    def crop_audio_window(self, original_mel, start_index):
        start_idx = int(80.0 * (start_index / float(self.video_fps)))
        end_idx = start_idx + self.mel_window_length
        return original_mel[:, start_idx:end_idx].unsqueeze(0)

    def get_frames(self, video_reader: VideoReader):
        total_num_frames = len(video_reader)

        start_idx = random.randint(self.num_frames // 2, total_num_frames - self.num_frames - self.num_frames // 2)
        frames_index = np.arange(start_idx, start_idx + self.num_frames, dtype=int)

        while True:
            wrong_start_idx = random.randint(0, total_num_frames - self.num_frames)
            if wrong_start_idx > start_idx - self.num_frames and wrong_start_idx < start_idx + self.num_frames:
                continue
            wrong_frames_index = np.arange(wrong_start_idx, wrong_start_idx + self.num_frames, dtype=int)
            break

        frames = video_reader.get_batch(frames_index).asnumpy()
        wrong_frames = video_reader.get_batch(wrong_frames_index).asnumpy()

        return frames, wrong_frames, start_idx

    def worker_init_fn(self, worker_id):
        # Initialize the face mesh object in each worker process,
        # because the face mesh object cannot be called in subprocesses
        self.worker_id = worker_id
        setattr(
            self,
            f"image_processor_{worker_id}",
            ImageProcessor(self.resolution, self.mask, mask_image=self.mask_image),
        )

    def __getitem__(self, idx):
        image_processor = getattr(self, f"image_processor_{self.worker_id}")
        while True:
            try:
                idx = random.randint(0, len(self) - 1)

                # Get video file path
                video_path = self.video_paths[idx]

                vr = VideoReader(video_path, ctx=cpu(self.worker_id))

                if len(vr) < 3 * self.num_frames:
                    continue

                continuous_frames, ref_frames, start_idx = self.get_frames(vr)

                if self.load_audio_data:
                    mel_cache_path = os.path.join(
                        self.audio_mel_cache_dir, os.path.basename(video_path).replace(".mp4", "_mel.pt")
                    )

                    if os.path.isfile(mel_cache_path):
                        try:
                            original_mel = torch.load(mel_cache_path)
                        except Exception as e:
                            print(f"{type(e).__name__} - {e} - {mel_cache_path}")
                            os.remove(mel_cache_path)
                            original_mel = self.read_audio(video_path)
                            torch.save(original_mel, mel_cache_path)
                    else:
                        original_mel = self.read_audio(video_path)
                        torch.save(original_mel, mel_cache_path)

                    mel = self.crop_audio_window(original_mel, start_idx)

                    if mel.shape[-1] != self.mel_window_length:
                        continue
                else:
                    mel = []

                gt, masked_gt, mask = image_processor.prepare_masks_and_masked_images(
                    continuous_frames, affine_transform=False
                )

                if self.mask == "fix_mask":
                    ref, _, _ = image_processor.prepare_masks_and_masked_images(ref_frames, affine_transform=False)
                else:
                    ref = image_processor.process_images(ref_frames)
                vr.seek(0)  # avoid memory leak
                break

            except Exception as e:  # Handle the exception of face not detcted
                print(f"{type(e).__name__} - {e} - {video_path}")
                if "vr" in locals():
                    vr.seek(0)  # avoid memory leak

        sample = dict(
            gt=gt,
            masked_gt=masked_gt,
            ref=ref,
            mel=mel,
            mask=mask,
            video_path=video_path,
            start_idx=start_idx,
        )

        return sample
