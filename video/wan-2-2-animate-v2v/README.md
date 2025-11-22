# WAN 2.2 Animate - Complete End-to-End Implementation

This is a production-ready implementation of Alibaba's WAN 2.2 Animate model with **integrated preprocessing** - no external scripts needed!

## 🎉 Key Features

- **✅ Raw Input**: Just provide a driving video + reference image - no preprocessing required!
- **✅ Integrated Preprocessing**: Automatic pose extraction, face detection, and masking built-in
- **✅ HuggingFace Downloads**: All models download automatically from HuggingFace Hub
- **✅ Two Modes**:
  - **Animation Mode**: Transfer character movements to your reference character
  - **Replacement Mode**: Replace character in video while preserving background
- **✅ Full Control**: Adjust resolution, FPS, pose retargeting, and all generation parameters
- **✅ Production Ready**: Follows inference.sh best practices

## 📋 Quick Start

### Input Requirements

**Just two files:**
1. **Driving Video** (MP4) - Person performing the actions you want to transfer
2. **Reference Image** (PNG/JPG) - The character you want to animate

**That's it!** No preprocessing, no skeleton extraction, no face cropping needed.

### Example Usage

```json
{
  "driving_video": {
    "path": "https://storage.googleapis.com/falserverless/example_inputs/wan_animate_input_video.mp4"
  },
  "reference_image": {
    "path": "https://storage.googleapis.com/falserverless/example_inputs/wan_animate_input_image.jpeg"
  },
  "mode": "animation",
  "sampling_steps": 20
}
```

### Run Locally

```bash
# Generate example input
infsh run

# Create your input.json (see example above)
# Then run with your inputs
infsh run input.json
```

## 🔧 Configuration Options

### Preprocessing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"animation"` | Mode: `"animation"` or `"replacement"` |
| `resolution_width` | 1280 | Target width (512-1920) |
| `resolution_height` | 720 | Target height (512-1920) |
| `fps` | 30 | Target FPS (-1 for original) |
| `retarget_flag` | false | Enable pose retargeting for different body proportions |
| `use_flux` | false | Use FLUX for better pose retargeting (slower) |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_len` | 77 | Frames per clip (must be 4n+1) |
| `refert_num` | 1 | Temporal guidance frames (1 or 5) |
| `shift` | 5.0 | Noise schedule shift |
| `sample_solver` | `"dpm++"` | Sampling solver (`"dpm++"` or `"unipc"`) |
| `sampling_steps` | 20 | Number of diffusion steps |
| `guide_scale` | 1.0 | Guidance scale for expression control |
| `seed` | -1 | Random seed (-1 for random) |

### Replacement Mode Parameters

Only used when `mode: "replacement"`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask_iterations` | 3 | Mask dilation iterations |
| `mask_kernel_size` | 7 | Mask dilation kernel size |
| `mask_w_subdivisions` | 1 | Mask width subdivisions |
| `mask_h_subdivisions` | 1 | Mask height subdivisions |

## 🎬 How It Works

### Pipeline Overview

```
1. INPUT
   - Driving Video (raw MP4)
   - Reference Image (PNG/JPG)

2. PREPROCESSING (Automatic)
   ├─ Pose Detection (YOLOv10m + VitPose)
   ├─ Face Extraction (crop & resize)
   ├─ Skeleton Drawing
   └─ [Optional] Masking (SAM2 for replacement mode)

3. GENERATION (WAN Animate 14B)
   └─ Diffusion-based video generation

4. OUTPUT
   └─ Animated Video (MP4)
```

### What Gets Preprocessed

The preprocessing pipeline automatically generates:

- **src_pose.mp4**: Skeleton/pose video extracted from driving video
- **src_face.mp4**: Cropped face regions for expression transfer
- **src_ref.png**: Processed reference image
- **src_bg.mp4** + **src_mask.mp4**: (Replacement mode only) Background and character mask

### Preprocessing Models

Downloaded automatically from HuggingFace:

- **YOLOv10m**: Person detection (~52MB)
- **VitPose-H**: Whole-body pose estimation (~600MB)
- **SAM2**: Segmentation for replacement mode (~800MB, optional)
- **FLUX.1-Kontext**: Image editing for better pose retargeting (~24GB, optional)

## 📦 Model Sizes

Total download size: **~60GB** (first run only)

- WAN Animate 14B: ~55GB
- Preprocessing models: ~5GB (+ 24GB for FLUX if using `use_flux: true`)

Models are cached by HuggingFace - subsequent runs are instant!

## 🎯 Use Cases

### Animation Mode
- Character animation from reference videos
- Dance video transfer
- Action sequence replication
- Expression and movement cloning

### Replacement Mode
- Replace character in existing video
- Preserve original background and environment
- Change character appearance while keeping actions

## ⚙️ System Requirements

### Minimum (Animation Mode)
- **GPU**: 40GB VRAM (A100, H100)
- **RAM**: 64GB
- **Storage**: 80GB for models + cache
- **Time**: ~2-5 minutes per video (depending on length)

### With FLUX (Better Quality)
- **GPU**: 48GB+ VRAM
- **Storage**: +24GB for FLUX model

## 🔬 Advanced Options

### Pose Retargeting

Enable when driving character has different body proportions than reference:

```json
{
  "retarget_flag": true,
  "use_flux": false
}
```

For even better results (but slower):

```json
{
  "retarget_flag": true,
  "use_flux": true
}
```

### Custom Prompts

Override default prompt (Chinese: "视频中的人在做动作"):

```json
{
  "input_prompt": "A person performing elegant dance movements",
  "n_prompt": "blurry, distorted, low quality"
}
```

### Guidance Scale

Use > 1.0 for more expression control (slower):

```json
{
  "guide_scale": 2.0
}
```

## 🐛 Troubleshooting

### "Preprocessing failed to generate files"
- Check input video has a visible person
- Try lower resolution or different FPS
- Enable `retarget_flag` for better pose detection

### Out of Memory
- Reduce `resolution_width` and `resolution_height`
- Lower `clip_len` (e.g., 49 or 37 instead of 77)
- Enable `offload_model: true`

### Poor Quality Results
- Enable `retarget_flag` if body proportions differ
- Try `use_flux: true` for better pose retargeting
- Increase `sampling_steps` (20 → 30+)
- Adjust `shift` parameter (try 3.0-7.0)

## 📚 References

- **WAN 2.2 Paper**: https://arxiv.org/abs/2503.20314
- **Official Repo**: https://github.com/Wan-Video/Wan2.2
- **HuggingFace Model**: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
- **WAN Animate Page**: https://humanaigc.github.io/wan-animate

## 📄 License

This implementation follows the WAN 2.2 license. See the [official repository](https://github.com/Wan-Video/Wan2.2) for details.

## 🙏 Credits

- **Alibaba Wan Team**: For creating WAN 2.2 Animate
- **inference.sh**: For the deployment platform
- This implementation integrates preprocessing directly for seamless end-to-end generation

---

**Built with ❤️ for the inference.sh platform**
