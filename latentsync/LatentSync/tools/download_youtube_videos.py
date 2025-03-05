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
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm

"""
To use this python file, first install yt-dlp by:

pip install yt-dlp==2024.5.27
"""


def download_video(video_url, video_path):
    get_video_channel_command = f"yt-dlp --print channel {video_url}"
    result = subprocess.run(get_video_channel_command, shell=True, capture_output=True, text=True)
    channel = result.stdout.strip()
    if channel in unwanted_channels:
        return
    download_video_command = f"yt-dlp -f bestvideo+bestaudio --skip-unavailable-fragments --merge-output-format mp4 '{video_url}' --output '{video_path}' --external-downloader aria2c --external-downloader-args '-x 16 -k 1M'"
    try:
        subprocess.run(download_video_command, shell=True)  # ignore_security_alert_wait_for_fix RCE
    except KeyboardInterrupt:
        print("Stopped")
        exit()
    except:
        print(f"Error downloading video {video_url}")


def download_videos(num_workers, video_urls, video_paths):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(download_video, video_urls, video_paths)


def read_video_urls(csv_file_path: str, language_column, video_url_column):
    video_urls = []
    print("Reading video urls...")
    df = pd.read_csv(csv_file_path, sep=",")
    for row in tqdm(df.itertuples(), total=len(df)):
        language = getattr(row, language_column)
        video_url = getattr(row, video_url_column)
        if "clip" in video_url:
            continue
        video_urls.append((language, video_url))
    return video_urls


def extract_vid(video_url):
    if "watch?v=" in video_url:  # ignore_security_alert_wait_for_fix RCE
        return video_url.split("watch?v=")[1][:11]
    elif "shorts/" in video_url:
        return video_url.split("shorts/")[1][:11]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1][:11]
    elif "&v=" in video_url:
        return video_url.split("&v=")[1][:11]
    else:
        print(f"Invalid video url: {video_url}")
        return None


def main(csv_file_path, language_column, video_url_column, output_dir, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    all_video_urls = read_video_urls(csv_file_path, language_column, video_url_column)

    video_paths = []
    video_urls = []

    print("Extracting vid...")
    for language, video_url in tqdm(all_video_urls):
        vid = extract_vid(video_url)
        if vid is None:
            continue
        video_path = os.path.join(output_dir, language.lower(), f"vid_{vid}.mp4")
        if os.path.isfile(video_path):
            continue
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_paths.append(video_path)
        video_urls.append(video_url)

    if len(video_paths) == 0:
        print("All videos have been downloaded")
        exit()
    else:
        print(f"Downloading {len(video_paths)} videos")

    download_videos(num_workers, video_urls, video_paths)


if __name__ == "__main__":
    csv_file_path = "dcc.csv"
    language_column = "video_language"
    video_url_column = "video_link"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/multilingual/raw"
    num_workers = 50

    unwanted_channels = ["TEDx Talks", "DaePyeong Mukbang", "Joeman"]

    main(csv_file_path, language_column, video_url_column, output_dir, num_workers)
