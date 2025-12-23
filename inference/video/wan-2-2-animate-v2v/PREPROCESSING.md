# WAN Animate Preprocessing Guide

## Important: Preprocessing Required!

**WAN Animate does NOT accept raw driving videos directly.** You must preprocess your input videos first using the WAN preprocessing pipeline.

### What is Preprocessing?

The preprocessing step extracts:
1. **`src_pose.mp4`** - Skeleton/pose information from the driving video
2. **`src_face.mp4`** - Cropped face regions from the driving video
3. **`src_bg.mp4`** + **`src_mask.mp4`** - (Optional, for character replacement mode)

### How to Preprocess

You have two options:

#### Option 1: Use Online Services

Most WAN Animate online services (like wan.video) handle preprocessing automatically:
- Upload your driving video + reference character image
- They preprocess it internally and generate the animation

#### Option 2: Run Preprocessing Locally

If you're running WAN locally, you need to preprocess first:

```bash
# Clone WAN 2.2 repository
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

# Install preprocessing dependencies
pip install -r requirements.txt
pip install -r requirements_animate.txt

# Download preprocessing models (pose detection, etc.)
# See: wan/modules/animate/preprocess/UserGuider.md

# Run preprocessing
python wan/modules/animate/preprocess/process_pipeline.py \
    --video_path /path/to/driving_video.mp4 \
    --refer_path /path/to/character_image.png \
    --save_path /path/to/output_dir \
    --ckpt_path /path/to/preprocessing_models \
    --mode animation \
    --retarget_flag  # Enable pose retargeting for different body proportions
```

### Preprocessing Modes

**Animation Mode** (default):
- Transfers character animation from driving video to reference character
- Supports pose retargeting for different body proportions
- Generates: `src_pose.mp4`, `src_face.mp4`, `src_ref.png`

**Replacement Mode** (with `--replace_flag`):
- Replaces character in driving video while preserving background
- Also generates: `src_bg.mp4`, `src_mask.mp4`

### Why No Preprocessing in This App?

Preprocessing requires additional models:
- YOLOv10m for person detection
- VitPose for pose estimation
- SAM2 for mask extraction
- FLUX.1 for image editing (optional)

These add ~10GB+ of model weights and complex dependencies. Most users prefer:
1. Using online services that handle preprocessing automatically
2. Running preprocessing separately as a one-time step

### For Developers

If you want to add preprocessing to this inference app:
1. Download preprocessing models from WAN 2.2 repo
2. Add preprocessing models to HuggingFace or local storage
3. Integrate `process_pipeline.py` into the `setup()` method
4. Add a preprocessing step before generation in `run()`

See the full preprocessing guide: https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md
