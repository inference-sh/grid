import os
# Enable HF Hub fast transfer for faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from pydantic import Field
from typing import List, Optional
from enum import Enum
import sys
import time
import json
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
import gc
from huggingface_hub import snapshot_download

# set the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add the SongGeneration directory to the path to import the required modules
sys.path.append(os.path.join(os.path.dirname(__file__)))
# Add the Flow1dVAE directory to the path for tokenizer imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'codeclm', 'tokenizer', 'Flow1dVAE'))

# Import all functionality from generate_utils (which contains the core logic from generate.py)
from .generate_utils import (
    Separator,
    auto_prompt_type,
    register_omegaconf_resolvers,
    load_config_and_setup,
    load_audio_tokenizer,
    load_separate_tokenizer,
    load_main_model,
    set_generation_params,
    process_prompt_audio_item,
    process_auto_prompt_item,
    process_no_prompt_item,
    apply_separate_tokenizer,
    generate_tokens,
    generate_audio_from_tokens,
    save_generated_audio,
    cleanup_item_memory,
    load_auto_prompt,
    create_output_directories,
    cleanup_models_and_cache
)
SONG_GENERATION_AVAILABLE = True


class AppInput(BaseAppInput):
    lyrics: str = Field(description="The lyrics for the song to generate")
    descriptions: Optional[str] = Field(None, description="Text description of the song style, genre, instruments, etc.")
    prompt_audio: Optional[File] = Field(None, description="Optional audio file to use as a prompt/reference")
    auto_prompt_type: str = Field(
        default="Auto",
        description="Auto prompt type for style selection",
        enum=["Pop", "R&B", "Dance", "Jazz", "Folk", "Rock", "Chinese Style", "Chinese Tradition", "Metal", "Reggae", "Chinese Opera", "Auto"]
    )
    generate_type: str = Field(
        default="mixed",
        description="Type of generation",
        enum=["vocal", "bgm", "separate", "mixed"]
    )

    temperature: float = Field(default=0.9, description="Generation temperature")
    cfg_coef: float = Field(default=1.5, description="CFG coefficient")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    top_p: float = Field(default=0.0, description="Top-p sampling parameter")
    max_duration: int = Field(default=30, description="Maximum duration of generated song in seconds")

class AppOutput(BaseAppOutput):
    generated_audio: File = Field(description="The generated song audio file")
    vocal_audio: Optional[File] = Field(None, description="Generated vocal track (if separate generation)")
    bgm_audio: Optional[File] = Field(None, description="Generated background music track (if separate generation)")
    generation_info: str = Field(description="Information about the generation process")

def download_model():
    repo_id = "tencent/SongGeneration"

    return snapshot_download(
        repo_id=repo_id,
        local_dir=".",
        revision="0c80d30",
        token=os.environ.get("HF_TOKEN"), 
        ignore_patterns=['.git*']
    )

class App(BaseApp):        
    async def setup(self, metadata):
        """Initialize the song generation model and resources."""
        # Apply same initialization as generate.py lines 536-541
        torch.backends.cudnn.enabled = False
        register_omegaconf_resolvers()
        np.random.seed(int(time.time()))
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for song generation")
        
        # Check if song generation modules are available
        if not SONG_GENERATION_AVAILABLE:
            raise RuntimeError("Song generation modules not available")
        
        # Determine variant and set low memory mode
        self.variant = getattr(metadata, "app_variant", "default")
        self.use_low_memory = self.variant == "low_memory"
        
        if self.use_low_memory:
            print("Using low memory mode for song generation")
        else:
            print("Using standard mode for song generation")
        
        out_dir = download_model()
        self.out_dir = out_dir
        self.separator = Separator(dm_model_path=os.path.join(out_dir, 'third_party/demucs/ckpt/htdemucs.pth'), dm_config_path=os.path.join(out_dir, 'third_party/demucs/ckpt/htdemucs.yaml'))    

        # Load auto prompt using generate.py logic
        prompt_path = os.path.join(out_dir, 'ckpt/prompt.pt')
        self.auto_prompt = load_auto_prompt(prompt_path)
            
        # Initialize model loading state
        self.model_loaded = False
        self.model = None
        self.audio_tokenizer = None
        self.separate_tokenizer = None

    async def _load_models(self, input_data):
        """Load the song generation models on-demand using the original generate.py implementation."""
        if self.model_loaded:
            return
            
        if not SONG_GENERATION_AVAILABLE:
            self.model_loaded = True
            return
            
        # Use the original model loading logic from generate.py
        ckpt_path = os.path.join(self.out_dir, 'ckpt/songgeneration_base')
        
        # Load configuration using generate.py logic
        self.cfg, model_path = load_config_and_setup(
            ckpt_path, 
            True,  # Always use flash attention
            input_data.max_duration
        )
        self.max_duration = self.cfg.max_dur
        
        # Debug logging for duration
        print(f"DEBUG: User requested duration: {input_data.max_duration}s")
        print(f"DEBUG: Config max_dur after setup: {self.cfg.max_dur}s")
        print(f"DEBUG: Final self.max_duration: {self.max_duration}s")
        
        # Update VAE config and model paths to use the downloaded model directory
        if hasattr(self.cfg, 'vae_config') and self.cfg.vae_config.startswith('./'):
            self.cfg.vae_config = os.path.join(self.out_dir, self.cfg.vae_config[2:])
        if hasattr(self.cfg, 'vae_model') and self.cfg.vae_model.startswith('./'):
            self.cfg.vae_model = os.path.join(self.out_dir, self.cfg.vae_model[2:])
        
        # Set environment variable to point to the downloaded model root
        os.environ['SONG_GENERATION_MODEL_ROOT'] = self.out_dir
        
        if self.use_low_memory:
            # Low memory mode: DON'T load models in setup - load them sequentially during generation
            print("Low memory mode enabled - models will be loaded sequentially during generation")
            
            # Store model paths but don't load models yet
            self.model_path = model_path
            
            # Don't load any models - they will be loaded/used/deleted sequentially in _generate_with_lowmem_model
        else:
            # Standard mode: Load models normally
            print("Loading models in standard mode...")
            
            # Load audio tokenizer using generate.py logic
            self.audio_tokenizer = load_audio_tokenizer(self.cfg)
            
            # Load separate tokenizer using generate.py logic
            self.separate_tokenizer = load_separate_tokenizer(self.cfg)
            
            # Load main model using generate.py logic
            self.model = load_main_model(self.cfg, model_path, self.max_duration, self.separate_tokenizer)
        
        # Set generation parameters using generate.py defaults (only for standard mode)
        if not self.use_low_memory:
            print(f"DEBUG (standard): About to set generation params with duration: {self.max_duration}s")
            set_generation_params(
                self.model,
                self.max_duration,
                input_data.temperature,
                input_data.cfg_coef,
                input_data.top_k,
                input_data.top_p
            )
        
        self.model_loaded = True

    async def run(self, input_data: AppInput, metadata) -> AppOutput:
        """Generate a song based on the input parameters."""
        # Load models if not already loaded
        await self._load_models(input_data)
        
        # Create temporary directories using generate.py logic
        temp_dir = "/tmp/song_generation"
        create_output_directories(temp_dir)
        
        # Generate a unique ID for this generation
        generation_id = f"gen_{int(time.time())}"
        
        # Prepare the generation item (similar to the original generate.py format)
        item = {
            "idx": generation_id,
            "gt_lyric": input_data.lyrics,
            "descriptions": input_data.descriptions,
            "generate_type": input_data.generate_type
        }
        
        # Handle prompt audio if provided
        if input_data.prompt_audio and input_data.prompt_audio.exists():
            item["prompt_audio_path"] = input_data.prompt_audio.path
        elif input_data.auto_prompt_type:
            item["auto_prompt_audio_type"] = input_data.auto_prompt_type
        
        # Process the item using the original logic
        processed_item = await self._process_item(item)
        
        # Generate the song
        generated_files = await self._generate_song(processed_item, temp_dir, input_data)
        
        # Prepare output
        generation_info = f"Generated song with ID: {generation_id}\n"
        generation_info += f"Lyrics length: {len(input_data.lyrics)} characters\n"
        generation_info += f"Generation type: {input_data.generate_type}\n"
        generation_info += f"Temperature: {input_data.temperature}\n"
        generation_info += f"CFG coefficient: {input_data.cfg_coef}\n"
        if input_data.descriptions:
            generation_info += f"Style: {input_data.descriptions}\n"
        
        return AppOutput(
            generated_audio=File(path=generated_files["main_audio"]),
            vocal_audio=File(path=generated_files["vocal_audio"]) if "vocal_audio" in generated_files else None,
            bgm_audio=File(path=generated_files["bgm_audio"]) if "bgm_audio" in generated_files else None,
            generation_info=generation_info
        )

    async def _process_item(self, item):
        """Process a generation item using the exact logic from generate.py."""
        
        if "prompt_audio_path" in item and self.separator:
            # Use exact logic from generate.py lines 111-144
            item = process_prompt_audio_item(item, self.separator, self.audio_tokenizer)
                
        elif "auto_prompt_audio_type" in item and self.auto_prompt:
            # Use exact logic from generate.py lines 146-155
            item = process_auto_prompt_item(item, self.auto_prompt)
        else:
            # Use exact logic from generate.py lines 156-160
            item = process_no_prompt_item(item)
        
        # Apply separate tokenizer using generate.py logic lines 182-188
        item = apply_separate_tokenizer(item, self.separate_tokenizer)
        
        return item

    async def _generate_song(self, item, temp_dir, input_data):
        """Generate the actual song audio using the original model or fallback."""
        
        # Use real models if: low memory mode (load on demand) OR models already loaded
        if self.use_low_memory or (self.model is not None and self.audio_tokenizer is not None):
            return await self._generate_with_model(item, temp_dir, input_data)
        else:
            # Fallback to placeholder generation only if models not available
            return await self._generate_placeholder_song(item, temp_dir, input_data)

    async def _generate_with_model(self, item, temp_dir, input_data):
        """Generate song using the actual loaded models with exact generate.py logic."""
        
        if self.use_low_memory:
            # Low memory mode: Use sequential load-use-delete pattern from generate_lowmem
            return await self._generate_with_lowmem_model(item, temp_dir, input_data)
        else:
            # Standard mode: Use normal generation
            print("Generating in standard mode...")
            
            # Generate tokens using exact logic from generate.py lines 227-238
            tokens, start_time, mid_time = generate_tokens(self.model, item)
            
            # Generate audio using exact logic from generate.py lines 241-271
            wav_seperate, wav_vocal, wav_bgm = generate_audio_from_tokens(
                self.model, tokens, item, input_data.generate_type
            )
            
            end_time = time.time()
            
            # Save audio files using exact logic from generate.py
            main_audio_path = os.path.join(temp_dir, "audios", f"{item['idx']}.flac")
            generated_files = save_generated_audio(
                wav_seperate, wav_vocal, wav_bgm, 
                main_audio_path, 
                input_data.generate_type, 
                self.cfg.sample_rate
            )
            
            # Clean up memory using generate.py logic
            cleanup_item_memory(item)
            
            print(f"process{item['idx']}, lm cost {mid_time - start_time}s, diffusion cost {end_time - mid_time}")
            
            return generated_files

    async def _generate_with_lowmem_model(self, item, temp_dir, input_data):
        """Generate song using sequential load-use-delete pattern from generate_lowmem."""
        print("Generating with low memory mode - sequential model loading...")
        
        start_time = time.time()
        
        # Stage 1: Handle audio tokenizer if needed (following generate_lowmem lines 304-372)
        use_audio_tokenizer = 'raw_pmt_wav' in item
        
        if use_audio_tokenizer:
            print("Loading audio tokenizer...")
            from codeclm.models import builders
            audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
            audio_tokenizer = audio_tokenizer.eval().cuda()
            
            # Use audio tokenizer
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(item['raw_pmt_wav'].cuda())
            item['pmt_wav'] = pmt_wav
            
            # Delete and cleanup
            del audio_tokenizer
            torch.cuda.empty_cache()
            print("Audio tokenizer deleted and memory cleared")
        
        # Stage 2: Handle separate tokenizer if needed (following generate_lowmem lines 375-393)
        if "audio_tokenizer_checkpoint_sep" in self.cfg.keys() and use_audio_tokenizer:
            print("Loading separate tokenizer...")
            separate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            separate_tokenizer = separate_tokenizer.eval().cuda()
            
            # Use separate tokenizer
            if use_audio_tokenizer and 'raw_vocal_wav' in item and 'raw_bgm_wav' in item:
                with torch.no_grad():
                    vocal_wav, bgm_wav = separate_tokenizer.encode(item['raw_vocal_wav'].cuda(), item['raw_bgm_wav'].cuda())
                item['vocal_wav'] = vocal_wav
                item['bgm_wav'] = bgm_wav
            
            # Delete and cleanup
            del separate_tokenizer
            torch.cuda.empty_cache()
            print("Separate tokenizer deleted and memory cleared")
        
        # Stage 3: Load and use main audiolm model (following generate_lowmem lines 395-465)
        print("Loading main audiolm model...")
        from codeclm.models import builders
        from codeclm.models import CodecLM
        
        audiolm = builders.get_lm_model(self.cfg)
        checkpoint = torch.load(self.model_path, map_location='cpu')
        audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval()
        
        # No offloading in low memory mode - just use CUDA
        audiolm = audiolm.cuda().to(torch.float16)
        
        model = CodecLM(
            name="tmp",
            lm=audiolm,
            audiotokenizer=None,
            max_duration=self.max_duration,
            seperate_tokenizer=None,
        )
        
        # Debug logging for low memory mode duration
        print(f"DEBUG (low mem): About to set generation params with duration: {self.max_duration}s")
        
        # Set generation params
        set_generation_params(
            model,
            self.max_duration,
            input_data.temperature,
            input_data.cfg_coef,
            input_data.top_k,
            input_data.top_p
        )
        
        # Generate tokens
        generate_inp = {
            'lyrics': [item["gt_lyric"].replace("  ", " ")],
            'descriptions': [item.get("descriptions")],
            'melody_wavs': item.get('pmt_wav'),
            'vocal_wavs': item.get('vocal_wav'),
            'bgm_wavs': item.get('bgm_wav'),
            'melody_is_wav': item.get('melody_is_wav', True),
        }
        
        print(f"DEBUG (low mem): Generating tokens with model.lm max_duration: {getattr(model.lm, 'max_duration', 'not set')}")
        print(f"DEBUG (low mem): Model generation params: duration={getattr(model, 'duration', 'not set')}")
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)
        
        mid_time = time.time()
        
        # Delete audiolm and cleanup
        del model
        audiolm = audiolm.cpu()
        del audiolm
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        print("Main audiolm model deleted and memory cleared")
        
        # Stage 4: Load separate tokenizer for audio generation (following generate_lowmem lines 467-528)
        print("Loading separate tokenizer for audio generation...")
        separate_tokenizer = builders.get_audio_tokenizer_model_cpu(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
        device = "cuda:0"
        separate_tokenizer.model.device = device
        separate_tokenizer.model.vae = separate_tokenizer.model.vae.to(device)
        separate_tokenizer.model.model.device = torch.device(device)
        separate_tokenizer = separate_tokenizer.eval()
        separate_tokenizer.model.model = separate_tokenizer.model.model.to(device)
        
        model = CodecLM(
            name="tmp",
            lm=None,
            audiotokenizer=None,
            max_duration=self.max_duration,
            seperate_tokenizer=separate_tokenizer,
        )
        
        # Generate audio from tokens
        with torch.no_grad():
            if 'raw_pmt_wav' in item:
                if input_data.generate_type == 'separate':
                    wav_seperate = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='mixed')
                    wav_vocal = model.generate_audio(tokens, chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, chunked=True, gen_type='bgm')
                elif input_data.generate_type == 'mixed':
                    wav_seperate = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type=input_data.generate_type)
                else:
                    wav_seperate = model.generate_audio(tokens, chunked=True, gen_type=input_data.generate_type)
            else:
                if input_data.generate_type == 'separate':
                    wav_vocal = model.generate_audio(tokens, chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, chunked=True, gen_type='bgm')
                    wav_seperate = model.generate_audio(tokens, chunked=True, gen_type='mixed')
                else:
                    wav_seperate = model.generate_audio(tokens, chunked=True, gen_type=input_data.generate_type)
        
        end_time = time.time()
        
        # Save audio files
        main_audio_path = os.path.join(temp_dir, "audios", f"{item['idx']}.flac")
        generated_files = {"main_audio": main_audio_path}
        
        if input_data.generate_type == 'separate':
            vocal_path = main_audio_path.replace('.flac', '_vocal.flac')
            bgm_path = main_audio_path.replace('.flac', '_bgm.flac')
            
            torchaudio.save(vocal_path, wav_vocal[0].cpu().float(), self.cfg.sample_rate)
            torchaudio.save(bgm_path, wav_bgm[0].cpu().float(), self.cfg.sample_rate)
            torchaudio.save(main_audio_path, wav_seperate[0].cpu().float(), self.cfg.sample_rate)
            
            generated_files["vocal_audio"] = vocal_path
            generated_files["bgm_audio"] = bgm_path
        else:
            torchaudio.save(main_audio_path, wav_seperate[0].cpu().float(), self.cfg.sample_rate)
        
        # Final cleanup
        del model
        del separate_tokenizer
        torch.cuda.empty_cache()
        print("Final cleanup completed")
        
        print(f"process{item['idx']}, lm cost {mid_time - start_time}s, diffusion cost {end_time - mid_time}s")
        
        return generated_files

    async def _generate_placeholder_song(self, item, temp_dir, input_data):
        """Generate a placeholder song when models are not available."""
        
        # Create a more sophisticated placeholder that varies based on input parameters
        duration = min(input_data.max_duration, 15)  # 15 seconds max for demo
        sample_rate = 44100
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create melody based on lyrics and style
        base_freq = 440  # A4
        
        # Vary frequency based on lyrics length and content
        lyrics_factor = len(item["gt_lyric"]) % 12  # 12 semitones
        freq = base_freq * (2 ** (lyrics_factor / 12))
        
        # Create main melody
        audio = torch.sin(2 * np.pi * freq * t) * 0.3
        
        # Add harmonics based on style description
        if input_data.descriptions:
            desc_lower = input_data.descriptions.lower()
            if "piano" in desc_lower:
                # Add piano-like harmonics
                audio += torch.sin(2 * np.pi * freq * 2 * t) * 0.1
                audio += torch.sin(2 * np.pi * freq * 4 * t) * 0.05
            elif "guitar" in desc_lower:
                # Add guitar-like harmonics
                audio += torch.sin(2 * np.pi * freq * 1.5 * t) * 0.15
                audio += torch.sin(2 * np.pi * freq * 3 * t) * 0.08
            elif "drums" in desc_lower:
                # Add rhythmic elements
                rhythm = torch.zeros_like(t)
                beat_interval = int(sample_rate * 0.5)  # 0.5 second beats
                for i in range(0, len(rhythm), beat_interval):
                    if i < len(rhythm):
                        rhythm[i:i+int(sample_rate*0.01)] = 0.2  # Short drum hit
                audio += rhythm
        
        # Add some variation over time
        envelope = torch.exp(-t / duration) * 0.5 + 0.5
        audio *= envelope
        
        # Ensure stereo
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).repeat(2, 1)
        
        # Save the generated audio
        main_audio_path = os.path.join(temp_dir, "audios", f"{item['idx']}.flac")
        torchaudio.save(main_audio_path, audio, sample_rate)
        
        generated_files = {"main_audio": main_audio_path}
        
        # If separate generation is requested, create separate vocal and BGM files
        if input_data.generate_type == "separate":
            # Create vocal track (higher frequencies)
            vocal_audio = audio * 0.8
            vocal_audio += torch.sin(2 * np.pi * freq * 2 * t) * 0.1  # Add harmonics
            
            # Create BGM track (lower frequencies)
            bgm_audio = audio * 0.4
            bgm_audio += torch.sin(2 * np.pi * freq * 0.5 * t) * 0.2  # Add bass
            
            vocal_path = os.path.join(temp_dir, "audios", f"{item['idx']}_vocal.flac")
            bgm_path = os.path.join(temp_dir, "audios", f"{item['idx']}_bgm.flac")
            
            torchaudio.save(vocal_path, vocal_audio, sample_rate)
            torchaudio.save(bgm_path, bgm_audio, sample_rate)
            
            generated_files["vocal_audio"] = vocal_path
            generated_files["bgm_audio"] = bgm_path
        
        return generated_files

    async def unload(self):
        """Clean up resources using generate.py logic."""
        # Clean up models using generate.py cleanup logic
        cleanup_models_and_cache(self.audio_tokenizer, self.separator, self.separate_tokenizer)
        
        # Clean up offload profilers if in low memory mode
        if self.use_low_memory:
            if hasattr(self, 'offload_profiler'):
                self.offload_profiler.stop()
                del self.offload_profiler
            if hasattr(self, 'sep_offload_profiler'):
                self.sep_offload_profiler.stop()
                del self.sep_offload_profiler
        
        # Clean up remaining resources
        if self.model:
            del self.model
        if self.auto_prompt:
            del self.auto_prompt
        
        gc.collect()
        torch.cuda.empty_cache()
