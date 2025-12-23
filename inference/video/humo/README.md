<div align="center">
<h1> HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning </h1>

<a href="https://arxiv.org/abs/2509.08519"><img src="https://img.shields.io/badge/arXiv%20paper-2509.08519-b31b1b.svg"></a>
<a href="https://phantom-video.github.io/HuMo/"><img src="https://img.shields.io/badge/Project_page-More_visualizations-green"></a>
<a href="https://huggingface.co/bytedance-research/HuMo"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>

[Liyang Chen](https://scholar.google.com/citations?user=jk6jWXgAAAAJ&hl)<sup> * </sup>, [Tianxiang Ma](https://tianxiangma.github.io/)<sup> * </sup>, [Jiawei Liu](https://scholar.google.com/citations?user=X21Fz-EAAAAJ), [Bingchuan Li](https://scholar.google.com/citations?user=ac5Se6QAAAAJ)<sup>&dagger;</sup>, <br>[Zhuowei Chen](https://scholar.google.com/citations?user=ow1jGJkAAAAJ), [Lijie Liu](https://liulj13.github.io/), [Xu He](https://scholar.google.com/citations?user=KMrFk2MAAAAJ&hl), [Gen Li](https://scholar.google.com/citations?user=wqA7EIoAAAAJ), [Qian He](https://scholar.google.com/citations?user=9rWWCgUAAAAJ), [Zhiyong Wu](https://scholar.google.com/citations?user=7Xl6KdkAAAAJ)<sup> § </sup><br>
<sup> * </sup>Equal contribution, <sup> &dagger; </sup>Project lead, <sup> § </sup>Corresponding author  
Tsinghua University | Intelligent Creation Team, ByteDance

</div>

<p align="center">
<img src="assets/teaser.png" width=95%>
<p>

## ✨ Key Features
HuMo is a unified, human-centric video generation framework designed to produce high-quality, fine-grained, and controllable human videos from multimodal inputs—including text, images, and audio. It supports strong text prompt following, consistent subject preservation, synchronized audio-driven motion.

> - **​​VideoGen from Text-Image**​​ - Customize character appearance, clothing, makeup, props, and scenes using text prompts combined with reference images.
> - **​​VideoGen from Text-Audio**​​ - Generate audio-synchronized videos solely from text and audio inputs, removing the need for image references and enabling greater creative freedom.
> - **​​VideoGen from Text-Image-Audio**​​ - Achieve the higher level of customization and control by combining text, image, and audio guidance.

## 📑 Todo List
- [x] Release Paper
- [x] Checkpoint of HuMo-17B
- [x] Inference Codes
  - [ ] Text-Image Input
  - [x] Text-Audio Input
  - [x] Text-Image-Audio Input
- [x] Multi-GPU Inference
- [ ] Prompts to Generate Demo of ***Faceless Thrones***
- [ ] Checkpoint of HuMo-1.7B
- [ ] Training Data

## ⚡️ Quickstart

### Installation
```
conda create -n humo python=3.11
conda activate humo
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install flash_attn==2.6.3
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

### Model Preparation
| Models       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| HuMo-17B      | 🤗 [Huggingface](https://huggingface.co/bytedance-research/HuMo/tree/main)   | Release before September 15
| HuMo-1.7B | 🤗 [Huggingface](https://huggingface.co/bytedance-research/HuMo/tree/main) | To be released soon
| Wan-2.1 | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | VAE & Text encoder
| Whisper-large-v3 |      🤗 [Huggingface](https://huggingface.co/openai/whisper-large-v3)          | Audio encoder
| Audio separator |      🤗 [Huggingface](https://huggingface.co/huangjackson/Kim_Vocal_2)          | Remove background noise (optional)

Download models using huggingface-cli:
``` sh
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./weights/Wan2.1-T2V-1.3B
huggingface-cli download bytedance-research/HuMo --local-dir ./weights/HuMo
huggingface-cli download openai/whisper-large-v3 --local-dir ./weights/whisper-large-v3
huggingface-cli download huangjackson/Kim_Vocal_2 --local-dir ./weights/audio_separator
```

### Run Multimodal-Condition-to-Video Generation

Our model is compatible with both 480P and 720P resolutions. 720P inference will achieve much better quality.
> Some tips
> - Please prepare your text, reference images and audio as described in [test_case.json](./examples/test_case.json).
> - We support Multi-GPU inference using FSDP + Sequence Parallel.
> - ​The model is trained on 97-frame videos at 25 FPS. Generating video longer than 97 frames may degrade the performance. We will provide a new checkpoint for longer generation.

#### Configure HuMo

HuMo’s behavior and output can be customized by modifying [generate.yaml](humo/configs/inference/generate.yaml) configuration file.  
The following parameters control generation length, video resolution, and how text, image, and audio inputs are balanced:

```yaml
generation:
  frames: <int>                 # Number of frames for the generated video.
  scale_a: <float>              # Strength of audio guidance. Higher = better audio-motion sync.
  scale_t: <float>              # Strength of text guidance. Higher = better adherence to text prompts.
  mode: "TA"                    # Input mode: "TA" for text+audio; "TIA" for text+image+audio.
  height: 720                   # Video height (e.g., 720 or 480).
  width: 1280                   # Video width (e.g., 1280 or 832).

diffusion:
  timesteps:
    sampling:
      steps: 50                 # Number of denoising steps. Lower (30–40) = faster generation.
```

#### 1. Text-Audio Input

``` sh
bash infer_ta.sh
```

#### 2. Text-Image-Audio Input

``` sh
bash infer_tia.sh
```

## Acknowledgements
Our work builds upon and is greatly inspired by several outstanding open-source projects, including [Phantom](https://github.com/Phantom-video/Phantom), [SeedVR](https://github.com/IceClear/SeedVR?tab=readme-ov-file), [MEMO](https://github.com/memoavatar/memo), [Hallo3](https://github.com/fudan-generative-vision/hallo3), [OpenHumanVid](https://github.com/fudan-generative-vision/OpenHumanVid), and [Whisper](https://github.com/openai/whisper). We sincerely thank the authors and contributors of these projects for generously sharing their excellent codes and ideas.

## ⭐ Citation

If HuMo is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our [paper](https://arxiv.org/abs/2509.08519).

### BibTeX
```bibtex
@misc{chen2025humo,
      title={HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning}, 
      author={Liyang Chen and Tianxiang Ma and Jiawei Liu and Bingchuan Li and Zhuowei Chen and Lijie Liu and Xu He and Gen Li and Qian He and Zhiyong Wu},
      year={2025},
      eprint={2509.08519},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.08519}, 
}
```

## 📧 Contact
If you have any comments or questions regarding this open-source project, please open a new issue or contact [Liyang Chen](lyangchen@outlook.com) and [Tianxiang Ma](https://tianxiangma.github.io/).