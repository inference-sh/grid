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
print("Loading models...")
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
pipe = pipe.to("cuda", torch.bfloat16)
print("Models loaded successfully!")

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
print("Image saved to output.jpg")