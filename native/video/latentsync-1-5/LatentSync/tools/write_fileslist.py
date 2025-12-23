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

from tqdm import tqdm
from latentsync.utils.util import gather_video_paths_recursively


def write_fileslist(fileslist_path):
    with open(fileslist_path, "w") as _:
        pass


def append_fileslist(fileslist_path, video_paths):
    with open(fileslist_path, "a") as f:
        for video_path in tqdm(video_paths):
            f.write(f"{video_path}\n")


def process_input_dir(fileslist_path, input_dir):
    print(f"Processing input dir: {input_dir}")
    video_paths = gather_video_paths_recursively(input_dir)
    append_fileslist(fileslist_path, video_paths)


if __name__ == "__main__":
    fileslist_path = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/fileslist/all_data_v6.txt"

    write_fileslist(fileslist_path)
    process_input_dir(fileslist_path, "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/high_visual_quality/train")
    process_input_dir(fileslist_path, "/mnt/bn/maliva-gen-ai-v2/chunyu.li/HDTF/high_visual_quality/train")
    process_input_dir(fileslist_path, "/mnt/bn/maliva-gen-ai-v2/chunyu.li/avatars/high_visual_quality/train")
    process_input_dir(fileslist_path, "/mnt/bn/maliva-gen-ai-v2/chunyu.li/multilingual/high_visual_quality")
    process_input_dir(fileslist_path, "/mnt/bn/maliva-gen-ai-v2/chunyu.li/celebv_text/high_visual_quality/train")
    process_input_dir(fileslist_path, "/mnt/bn/maliva-gen-ai-v2/chunyu.li/youtube/high_visual_quality")
