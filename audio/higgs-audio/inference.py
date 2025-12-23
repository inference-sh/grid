import sys
import os
# Add higgs-audio project to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'higgs-audio'))

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field, BaseModel
from typing import Optional, List
import torch
import soundfile as sf
import copy
import re
import logging
from loguru import logger

# Higgs Audio imports
from boson_multimodal.model.higgs_audio import HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache
from dataclasses import asdict
import tqdm

class AppInput(BaseAppInput):
    transcript: str = Field(description="The text to convert to speech")
    temperature: float = Field(default=1.0, description="The value used to module the next token probabilities")
    top_k: int = Field(default=50, description="The number of highest probability vocabulary tokens to keep for top-k-filtering")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    max_new_tokens: int = Field(default=2048, description="The maximum number of new tokens to generate")
    seed: Optional[int] = Field(default=None, description="Random seed for generation")
    ref_audio: Optional[str] = Field(default=None, description="Voice reference audio name (e.g., 'belinda', 'broom_salesman') or comma-separated for multi-speaker")
    scene_prompt: Optional[str] = Field(default=None, description="Scene description prompt for context")
    chunk_method: Optional[str] = Field(default=None, description="Chunking method: 'speaker', 'word', or None")
    chunk_max_word_num: int = Field(default=200, description="Maximum words per chunk when using word chunking")
    chunk_max_num_turns: int = Field(default=1, description="Maximum turns per chunk when using speaker chunking")
    generation_chunk_buffer_size: Optional[int] = Field(default=None, description="Maximum chunks to keep in buffer")
    ras_win_len: int = Field(default=7, description="RAS sampling window length (0 to disable)")
    ras_win_max_num_repeat: int = Field(default=2, description="Maximum RAS window repeats")
    ref_audio_in_system_message: bool = Field(default=False, description="Include reference audio description in system message")

class AppOutput(BaseAppOutput):
    audio_output: File = Field(description="The generated audio file")

# Constants
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"
MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

def normalize_chinese_punctuation(text):
    """Convert Chinese (full-width) punctuation marks to English (half-width) equivalents."""
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        """: '"', """: '"', "'": "'", "'": "'", "、": ",", "—": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def prepare_chunk_text(text, chunk_method=None, chunk_max_word_num=100, chunk_max_num_turns=1):
    """Chunk the text into smaller pieces."""
    if chunk_method is None:
        return [text]
    elif chunk_method == "word":
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_max_word_num):
            chunk = " ".join(words[i:i + chunk_max_word_num])
            chunks.append(chunk)
        return chunks
    elif chunk_method == "speaker":
        # Split by speaker tags
        pattern = re.compile(r"(\[SPEAKER\d+\])")
        parts = pattern.split(text)
        chunks = []
        current_chunk = ""
        turn_count = 0
        
        for part in parts:
            if pattern.match(part):
                if current_chunk and turn_count >= chunk_max_num_turns:
                    chunks.append(current_chunk.strip())
                    current_chunk = part
                    turn_count = 1
                else:
                    current_chunk += part
                    if current_chunk.strip():
                        turn_count += 1
            else:
                current_chunk += part
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    else:
        return [text]

def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
    """Prepare the context messages and audio IDs for generation."""
    messages = []
    audio_ids = []
    
    # Handle reference audio
    if ref_audio:
        if "," in ref_audio:
            # Multi-speaker scenario
            ref_audios = [name.strip() for name in ref_audio.split(",")]
        else:
            ref_audios = [ref_audio]
        
        # For simplicity, we'll use a basic system message
        # In a real implementation, you'd load actual audio files here
        if ref_audio_in_system_message:
            speaker_desc_l = []
            for i, audio_name in enumerate(ref_audios):
                if i < len(speaker_tags):
                    speaker_desc_l.append(f"{speaker_tags[i]}: {audio_name} voice")
            
            if speaker_desc_l:
                speaker_desc = "\n".join(speaker_desc_l)
                scene_desc_l = []
                if scene_prompt:
                    scene_desc_l.append(scene_prompt)
                scene_desc_l.append(speaker_desc)
                scene_desc = "\n\n".join(scene_desc_l)
                
                system_message = Message(
                    role="system",
                    content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
                )
            else:
                system_message_l = ["Generate audio following instruction."]
                if scene_prompt:
                    system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
                system_message = Message(
                    role="system",
                    content="\n\n".join(system_message_l),
                )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(
                role="system",
                content="\n\n".join(system_message_l),
            )
    else:
        system_message_l = ["Generate audio following instruction."]
        if scene_prompt:
            system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
        system_message = Message(
            role="system",
            content="\n\n".join(system_message_l),
        )
    
    if system_message:
        messages.insert(0, system_message)
    
    return messages, audio_ids

class App(BaseApp):
    async def setup(self, metadata):
        """Initialize the Higgs Audio model and resources."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model configuration
        self.model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        self.audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        
        logger.info(f"Loading Higgs Audio model from {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load audio tokenizer
        self.audio_tokenizer = load_higgs_audio_tokenizer(self.audio_tokenizer_path, device=self.device)
        
        # Load model
        self.model = HiggsAudioModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        
        # Load tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.config = AutoConfig.from_pretrained(self.model_path)
        
        # Initialize collator
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self.config.audio_in_token_idx,
            audio_out_token_id=self.config.audio_out_token_idx,
            audio_stream_bos_id=self.config.audio_stream_bos_id,
            audio_stream_eos_id=self.config.audio_stream_eos_id,
            encode_whisper_embed=self.config.encode_whisper_embed,
            pad_token_id=self.config.pad_token_id,
            return_audio_in_tokens=self.config.encode_audio_in_tokens,
            use_delay_pattern=self.config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self.config.audio_num_codebooks,
        )
        
        # Initialize KV cache for faster generation
        self.kv_cache_lengths = [1024, 4096, 8192]
        self._init_static_kv_cache()
        
        logger.info("Higgs Audio model loaded successfully")

    def _init_static_kv_cache(self):
        """Initialize static KV cache for faster generation."""
        cache_config = copy.deepcopy(self.model.config.text_config)
        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(self.kv_cache_lengths)
        }
        
        # Capture CUDA graphs for each KV cache length
        if "cuda" in self.device:
            logger.info("Capturing CUDA graphs for KV cache optimization")
            self.model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        """Reset KV caches for new generation."""
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate audio from text using Higgs Audio model."""
        logger.info(f"Generating audio for transcript: '{input_data.transcript[:100]}...'")
        
        # Prepare transcript
        transcript = input_data.transcript
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(transcript)))
        
        # Basic text normalization
        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit")
        transcript = transcript.replace("°C", " degrees Celsius")
        
        # Handle special audio tags
        for tag, replacement in [
            ("[laugh]", "<SE>[Laughter]</SE>"),
            ("[humming start]", "<SE>[Humming]</SE>"),
            ("[humming end]", "<SE_e>[Humming]</SE_e>"),
            ("[music start]", "<SE_s>[Music]</SE_s>"),
            ("[music end]", "<SE_e>[Music]</SE_e>"),
            ("[music]", "<SE>[Music]</SE>"),
            ("[sing start]", "<SE_s>[Singing]</SE_s>"),
            ("[sing end]", "<SE_e>[Singing]</SE_e>"),
            ("[applause]", "<SE>[Applause]</SE>"),
            ("[cheering]", "<SE>[Cheering]</SE>"),
            ("[cough]", "<SE>[Cough]</SE>"),
        ]:
            transcript = transcript.replace(tag, replacement)
        
        # Clean up whitespace
        lines = transcript.split("\n")
        transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
        transcript = transcript.strip()
        
        # Ensure proper punctuation
        if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
            transcript += "."
        
        # Prepare generation context
        messages, audio_ids = prepare_generation_context(
            scene_prompt=input_data.scene_prompt,
            ref_audio=input_data.ref_audio,
            ref_audio_in_system_message=input_data.ref_audio_in_system_message,
            audio_tokenizer=self.audio_tokenizer,
            speaker_tags=speaker_tags,
        )
        
        # Prepare text chunks
        chunked_text = prepare_chunk_text(
            transcript,
            chunk_method=input_data.chunk_method,
            chunk_max_word_num=input_data.chunk_max_word_num,
            chunk_max_num_turns=input_data.chunk_max_num_turns,
        )
        
        logger.info(f"Processing {len(chunked_text)} text chunks")
        
        # Generate audio
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        
        ras_win_len = input_data.ras_win_len if input_data.ras_win_len > 0 else None
        
        for idx, chunk_text in enumerate(chunked_text):
            logger.info(f"Processing chunk {idx + 1}/{len(chunked_text)}")
            
            generation_messages.append(
                Message(
                    role="user",
                    content=chunk_text,
                )
            )
            
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self.tokenizer)
            postfix = self.tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)
            
            context_audio_ids = audio_ids + generated_audio_ids
            
            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )
            
            batch_data = self.collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self.device)
            
            # Reset KV cache
            self._prepare_kv_caches()
            
            # Generate audio tokens
            with torch.inference_mode():
                outputs = self.model.generate(
                    **batch,
                    max_new_tokens=input_data.max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=input_data.temperature,
                    top_k=input_data.top_k,
                    top_p=input_data.top_p,
                    past_key_values_buckets=self.kv_caches,
                    ras_win_len=ras_win_len,
                    ras_win_max_num_repeat=input_data.ras_win_max_num_repeat,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                    tokenizer=self.tokenizer,
                    seed=input_data.seed,
                )
            
            # Process generated audio tokens
            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self.config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self.audio_tokenizer.codebook_size - 1)[:, 1:-1])
            
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)
            
            generation_messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=""),
                )
            )
            
            # Apply buffer size limit
            if input_data.generation_chunk_buffer_size is not None and len(generated_audio_ids) > input_data.generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-input_data.generation_chunk_buffer_size:]
        
        # Concatenate all audio chunks and decode
        if audio_out_ids_l:
            concat_audio_ids = torch.concat(audio_out_ids_l, dim=1)
            logger.info(f"Decoding audio tokens with shape: {concat_audio_ids.shape}")
            
            # Decode audio tokens to waveform - need to add batch dimension and detach gradients
            with torch.no_grad():
                concat_wv = self.audio_tokenizer.decode(concat_audio_ids.detach().unsqueeze(0))[0, 0]
            if isinstance(concat_wv, torch.Tensor):
                concat_wv = concat_wv.detach().cpu().numpy()
            
            # Ensure mono audio
            if len(concat_wv.shape) > 1:
                concat_wv = concat_wv.mean(axis=0)
            
            # Save audio file
            output_path = "/tmp/generated_audio.wav"
            sf.write(output_path, concat_wv, sr)
            
            logger.info(f"Audio generated successfully, saved to {output_path}")
            logger.info(f"Audio duration: {len(concat_wv) / sr:.2f} seconds")
            
            return AppOutput(audio_output=File(path=output_path))
        else:
            raise RuntimeError("No audio was generated")
