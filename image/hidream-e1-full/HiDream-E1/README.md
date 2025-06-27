# HiDream-E1

![HiDream-E1 Demo](assets/demo.jpg)

HiDream-E1 is an image editing model built on [HiDream-I1](https://github.com/HiDream-ai/HiDream-I1).

<!-- ![Overview](demo.jpg) -->
<span style="color: #FF5733; font-weight: bold">For more features and to experience the full capabilities of our product, please visit [https://vivago.ai/](https://vivago.ai/).</span>

## Project Updates
- 📝 **May 28, 2025**: We've released our technical report [HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer](https://arxiv.org/abs/2505.22705).
- 🚀 **April 28, 2025**: We've open-sourced the instruction-based image editing model **HiDream-E1**. 


## Models

We offer the full version of HiDream-E1. For more information about the models, please refer to the link under Usage.

| Name            | Script                                             | Inference Steps | Resolution | HuggingFace repo       |
| --------------- | -------------------------------------------------- | --------------- | ---------- | ---------------------- |
| HiDream-E1-Full | [inference.py](./inference.py)                     | 28              | 768x768    | 🤗 [HiDream-E1-Full](https://huggingface.co/HiDream-ai/HiDream-E1-Full)  |
> [!NOTE]
> The code and model are under development and will be updated frequently.


## Quick Start
Please make sure you have installed [Flash Attention](https://github.com/Dao-AILab/flash-attention) and latest [Diffusers](https://github.com/huggingface/diffusers.git). We recommend CUDA versions 12.4 for the manual installation.

```sh
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
pip install -U git+https://github.com/huggingface/diffusers.git
```

Then you can run the inference scripts to generate images:

``` python 
python ./inference.py
```

Alternatively, you can use the model in your own code:

```python
import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from PIL import Image
from peft import LoraConfig
from huggingface_hub import hf_hub_download
from diffusers import HiDreamImageTransformer2DModel
from instruction_refinement import refine_instruction
from safetensors.torch import load_file

# Set to True to enable instruction refinement and transformer model
ENABLE_REFINE = True

# Load models
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

# Configure transformer model if refinement is enabled
transformer = None
reload_keys = None
if ENABLE_REFINE:
    transformer = HiDreamImageTransformer2DModel.from_pretrained("HiDream-ai/HiDream-I1-Full", subfolder="transformer")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["to_k", "to_q", "to_v", "to_out", "to_k_t", "to_q_t", "to_v_t", "to_out_t", "w1", "w2", "w3", "final_layer.linear"],
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)
    transformer.max_seq = 4608
    lora_ckpt_path = hf_hub_download(repo_id="HiDream-ai/HiDream-E1-Full", filename="HiDream-E1-Full.safetensors")
    lora_ckpt = load_file(lora_ckpt_path, device="cuda")
    src_state_dict = transformer.state_dict()
    reload_keys = [k for k in lora_ckpt if "lora" not in k]
    reload_keys = {
        "editing": {k: v for k, v in lora_ckpt.items() if k in reload_keys},
        "refine": {k: v for k, v in src_state_dict.items() if k in reload_keys},
    }
    info = transformer.load_state_dict(lora_ckpt, strict=False)
    assert len(info.unexpected_keys) == 0

# Initialize pipeline
if ENABLE_REFINE:
    pipe = HiDreamImageEditingPipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Full",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
        transformer=transformer,
    )
else:
    pipe = HiDreamImageEditingPipeline.from_pretrained(
        "HiDream-ai/HiDream-E1-Full",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    )

# Load and preprocess test image
test_image = Image.open("assets/test_1.png")
original_width, original_height = test_image.size
test_image = test_image.resize((768, 768))

# Define instruction
instruction = 'Convert the image into a Ghibli style.'

# Refine instruction if enabled
refined_instruction = refine_instruction(src_image=test_image, src_instruction=instruction)
print(f"Original instruction: {instruction}")
print(f"Refined instruction: {refined_instruction}")

# Move pipeline to GPU
pipe = pipe.to("cuda", torch.bfloat16)

# Generate image
image = pipe(
    prompt=refined_instruction,
    negative_prompt="low resolution, blur",
    image=test_image,
    guidance_scale=5.0,
    image_guidance_scale=4.0,
    num_inference_steps=28,
    generator=torch.Generator("cuda").manual_seed(3),
    refine_strength=0.3 if ENABLE_REFINE else 0.0,
    reload_keys=reload_keys,
).images[0]

# Resize back to original dimensions and save
image = image.resize((original_width, original_height))
image.save("output.jpg")
```

> [!NOTE]
> The inference script will try to automatically download `meta-llama/Llama-3.1-8B-Instruct` model files. You need to [agree to the license of the Llama model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on your HuggingFace account and login using `huggingface-cli login` in order to use the automatic downloader.


> [!NOTE]
> The model accepts instructions in the following format:
> ```
> Editing Instruction: {instruction}. Target Image Description: {description}
> ```
> 
> Example:
> ```
> Editing Instruction: Convert the image into a Ghibli style. Target Image Description: A person in a light pink t-shirt with short dark hair, depicted in a Ghibli style against a plain background.
> ```
> 
> To refine your instructions, use the provided script:
> ```bash
> python ./instruction_refinement.py --src_image ./test.jpeg --src_instruction "convert the image into a Ghibli style"
> ```
> 
> The instruction refinement script requires a VLM API key - you can either run vllm locally or use OpenAI's API.

> [!NOTE]
> We add a refine_strength parameter to the pipeline to control the balance between editing and refinement stages. During the first (1 - refine_strength) portion of denoising steps, the model performs the main editing operation. The remaining refine_strength portion of steps uses HiDream-I1-Full for img2img refinement to enhance the final result. Set refine_strength to 0.0 to disable refinement.

## Gradio Demo

We also provide a Gradio demo for interactive image editing. You can run the demo with:

``` python
python gradio_demo.py 
```
<!-- 
## Examples

Below are demonstration examples of HiDream-E1's capabilities:

![Example Results](case.jpg) -->


## Evaluation Metrics

**Evaluation results on EmuEdit and ReasonEdit Benchmarks. Higher is better.**

| Model              | EmuEdit Global | EmuEdit Add  | EmuEdit Text | EmuEdit BG   | EmuEdit Color | EmuEdit Style | EmuEdit Remove | EmuEdit Local | EmuEdit Average | ReasonEdit |
|--------------------|----------------|--------------|--------------|--------------|---------------|---------------|----------------|---------------|-----------------|------------|
| OmniGen            | 1.37           | 2.09         | 2.31         | 0.66         | 4.26          | 2.36          | 4.73           | 2.10          | 2.67            | 7.36       |
| MagicBrush         | 4.06           | 3.54         | 0.55         | 3.26         | 3.83          | 2.07          | 2.70           | 3.28          | 2.81            | 1.75       |
| UltraEdit          | 5.31           | 5.19         | 1.50         | 4.33         | 4.50          | 5.71          | 2.63           | 4.58          | 4.07            | 2.89       |
| Gemini-2.0-Flash   | 4.87           | **7.71** | 6.30         | **5.10** | 7.30          | 3.33          | 5.94           | 6.29          | 5.99            | 6.95       |
| HiDream-E1         | **5.32** | 6.98         | **6.45** | 5.01         | **7.57** | **6.49** | **5.99** | **6.35** | **6.40** | **7.54** |

## License

The code in this repository and the HiDream-E1 models are licensed under [MIT License](./LICENSE).

## Citation

```bibtex
@article{hidreami1technicalreport,
  title={HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer},
  author={Cai, Qi and Chen, Jingwen and Chen, Yang and Li, Yehao and Long, Fuchen and Pan, Yingwei and Qiu, Zhaofan and Zhang, Yiheng and Gao, Fengbin and Xu, Peihan and others},
  journal={arXiv preprint arXiv:2505.22705},
  year={2025}
}
```
