import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from instruction_refinement import refine_instruction
import gradio as gr
from peft import LoraConfig
from huggingface_hub import hf_hub_download
from diffusers import HiDreamImageTransformer2DModel
from safetensors.torch import load_file
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
else:
    transformer = None
    reload_keys = None
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

def process_image(input_image, instruction, seed, guidance_scale, image_guidance_scale, steps, refine_instruction_option, refine_strength):
    
    original_width, original_height = input_image.size
    
    # Resize the input image to 768x768
    input_image = input_image.resize((768, 768))
    
    # Refine the instruction if option is selected
    if refine_instruction_option:
        refined_instruction = refine_instruction(src_image=input_image, src_instruction=instruction)
    else:
        refined_instruction = instruction
    
    # Generate image
    generator = torch.Generator("cuda").manual_seed(seed)
    if refine_strength > 0 and not ENABLE_REFINE:
        print("Refine will not be used since ENABLE_REFINE is False")

    result = pipe(
        prompt=refined_instruction,
        negative_prompt="low resolution, blur",
        image=input_image,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=steps,
        generator=generator,
        refine_strength=refine_strength,
        reload_keys=reload_keys,
    ).images[0]
    
    # Resize the result to the original size
    result = result.resize((original_width, original_height))
    
    return result, refined_instruction

# Create Gradio interface
with gr.Blocks(title="HiDream Image Editor") as demo:
    gr.Markdown("# HiDream Image Editing")
    gr.Markdown("Upload an image and provide an instruction to edit it using the HiDream model.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            instruction = gr.Textbox(label="Editing Instruction", placeholder="e.g., convert the image into a Ghibli style")
            refine_instruction_option = gr.Checkbox(label="Refine Instruction", value=True, info="Use VLM to refine your instruction")
            
            with gr.Accordion("Advanced Settings", open=True):
                seed = gr.Slider(minimum=0, maximum=10000, step=1, value=3, label="Seed")
                gr.Markdown("*Note: You might need to try different seeds to get satisfying results.*")
                guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=5.0, label="Instruction Following Strength")
                image_guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="Image Preservation Strength")
                gr.Markdown("*Note: For style changes, use higher image preservation strength (e.g., 3.0-4.0). For local edits like adding, deleting, replacing elements, use lower image preservation strength (e.g., 2.0-3.0).*")
                steps = gr.Slider(minimum=10, maximum=50, step=1, value=28, label="Steps")
                refine_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.3, label="Refine Strength")
            submit_btn = gr.Button("Generate")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Output Image", format="png")
            refined_instruction_text = gr.Textbox(label="Refined Instruction", interactive=False)
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, instruction, seed, guidance_scale, image_guidance_scale, steps, refine_instruction_option, refine_strength],
        outputs=[output_image, refined_instruction_text]
    )
    
    gr.Examples(
        examples=[
            ["assets/test_1.png", "convert the image into a Ghibli style",3, 5, 4, 28, True, 0.3],
            ["assets/test_1.png", "change the image into Disney Pixar style",3, 5, 4, 28, True, 0.3],
            ["assets/test_1.png", "turn to sketch style",3, 5, 4, 28, True, 0.3],
            ["assets/test_1.png", "add a sunglasses to the girl",3, 5, 2, 28, True, 0.3],
            ["assets/test_1.png", "change the background to a sunset",3, 5, 2, 28, True, 0.3],
            ["assets/test_2.jpg", "convert this image into a ink sketch image",3, 5, 2, 28, True, 0.3],
            ["assets/test_2.jpg", "add butterfly",3, 5, 2, 28, True, 0.3],
            ["assets/test_2.jpg", "remove the wooden sign",3, 5, 2, 28, True, 0.3],
        ],
        fn=process_image,
        inputs=[input_image, instruction, seed, guidance_scale, image_guidance_scale, steps, refine_instruction_option, refine_strength],
        outputs=[output_image, refined_instruction_text],
        cache_examples=True,
        cache_mode="lazy",
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
