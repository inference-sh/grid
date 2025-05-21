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

import mediapipe as mp
import cv2
from decord import VideoReader
from einops import rearrange
import os
import numpy as np
import torch
import tqdm
from eval.fvd import compute_our_fvd


class FVD:
    def __init__(self, resolution=(224, 224)):
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.resolution = resolution

    def detect_face(self, image):
        height, width = image.shape[:2]
        # Process the image and detect faces.
        results = self.face_detector.process(image)

        if not results.detections:  # Face not detected
            raise Exception("Face not detected")

        detection = results.detections[0]  # Only use the first face in the image
        bounding_box = detection.location_data.relative_bounding_box
        xmin = int(bounding_box.xmin * width)
        ymin = int(bounding_box.ymin * height)
        face_width = int(bounding_box.width * width)
        face_height = int(bounding_box.height * height)

        # Crop the image to the bounding box.
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmin + face_width)
        ymax = min(height, ymin + face_height)
        image = image[ymin:ymax, xmin:xmax]

        return image

    def detect_video(self, video_path, real: bool = True):
        vr = VideoReader(video_path)
        video_frames = vr[20:36].asnumpy()  # Use one frame per second
        vr.seek(0)  # avoid memory leak
        faces = []
        for frame in video_frames:
            face = self.detect_face(frame)
            face = cv2.resize(face, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            faces.append(face)

        if len(faces) != 16:
            return None
        faces = np.stack(faces, axis=0)  # (f, h, w, c)
        faces = torch.from_numpy(faces)
        return faces


def eval_fvd(real_videos_dir, fake_videos_dir):
    fvd = FVD()
    real_features_list = []
    fake_features_list = []
    for file in tqdm.tqdm(os.listdir(fake_videos_dir)):
        if file.endswith(".mp4"):
            real_video_path = os.path.join(real_videos_dir, file.replace("_out.mp4", ".mp4"))
            fake_video_path = os.path.join(fake_videos_dir, file)
            real_features = fvd.detect_video(real_video_path, real=True)
            fake_features = fvd.detect_video(fake_video_path, real=False)
            if real_features is None or fake_features is None:
                continue
            real_features_list.append(real_features)
            fake_features_list.append(fake_features)

    real_features = torch.stack(real_features_list) / 255.0
    fake_features = torch.stack(fake_features_list) / 255.0
    print(compute_our_fvd(real_features, fake_features, device="cpu"))


if __name__ == "__main__":
    real_videos_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/segmented/cross"
    fake_videos_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/segmented/latentsync_cross"

    eval_fvd(real_videos_dir, fake_videos_dir)
