# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Inference codes adapted from [SeedVR]
# https://github.com/ByteDance-Seed/SeedVR/blob/main/projects/inference_seedvr2_7b.py

from sys import argv
import sys

path_to_insert = "humo"
if path_to_insert not in sys.path:
    sys.path.insert(0, path_to_insert)

from common.config import load_config, create_object

# Load config.
config = load_config(argv[1], argv[2:])

runner = create_object(config)
runner.entrypoint()
