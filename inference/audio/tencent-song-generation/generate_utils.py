"""
Utility functions extracted from generate.py to be reused by inference.py
These functions contain the core logic from generate.py without the command-line interface
"""

import os
import sys
import time
import json
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
import gc
from codeclm.models import builders
from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from third_party.demucs.models.pretrained import get_model_from_yaml

# Copy from generate.py to avoid import issues
auto_prompt_type = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']

class Separator:
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def load_audio(self, f):
        a, fs = torchaudio.load(f)
        if (fs != 48000):
            a = torchaudio.functional.resample(a, fs, 48000)
        if a.shape[-1] >= 48000*10:
            a = a[..., :48000*10]
        return a[:, 0:48000*10]
    
    def run(self, audio_path, output_dir='tmp', ext=".flac"):
        os.makedirs(output_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []

        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 1:  # 4
            vocal_path = output_paths[0]
        else:
            drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
            for path in [drums_path, bass_path, other_path]:
                os.remove(path)
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        bgm_audio = full_audio - vocal_audio
        return full_audio, vocal_audio, bgm_audio


def register_omegaconf_resolvers():
    """Register OmegaConf resolvers with proper type handling"""
    
    def safe_eval(x):
        """Eval resolver that ensures proper types"""
        result = eval(x)
        return result
    
    def safe_concat(*x):
        """Concat resolver that preserves types"""
        return [xxx for xx in x for xxx in xx]
    
    def safe_get_fname():
        """Get filename resolver"""
        return os.path.splitext(os.path.basename(sys.argv[1]))[0] if len(sys.argv) > 1 else "default"
    
    def safe_load_yaml(x):
        """Load YAML resolver that ensures proper types"""
        loaded = OmegaConf.load(x)
        # Convert to container to resolve all interpolations and ensure proper types
        container = OmegaConf.to_container(loaded, resolve=True)
        return list(container) if isinstance(container, (list, tuple)) else [container]
    
    try:
        # Try the newer method first (OmegaConf >= 2.1)
        OmegaConf.register_new_resolver("eval", safe_eval)
        OmegaConf.register_new_resolver("concat", safe_concat)
        OmegaConf.register_new_resolver("get_fname", safe_get_fname)
        OmegaConf.register_new_resolver("load_yaml", safe_load_yaml)
    except AttributeError:
        # Fallback to older method (OmegaConf < 2.1)
        OmegaConf.register_resolver("eval", safe_eval)
        OmegaConf.register_resolver("concat", safe_concat)
        OmegaConf.register_resolver("get_fname", safe_get_fname)
        OmegaConf.register_resolver("load_yaml", safe_load_yaml)


def load_auto_prompt(prompt_path='ckpt/prompt.pt'):
    """Load auto prompt using the same logic as generate.py line 98"""
    if os.path.exists(prompt_path):
        return torch.load(prompt_path)
    return None


def load_config_and_setup(ckpt_path, use_flash_attn=False, max_duration=None):
    """Load configuration and setup paths, extracted from generate.py logic"""
    cfg_path = os.path.join(ckpt_path, 'config.yaml')
    model_path = os.path.join(ckpt_path, 'model.pt')
    
    if not os.path.exists(cfg_path) or not os.path.exists(model_path):
        raise RuntimeError(f"Model checkpoint not found at {ckpt_path}")
    
    cfg = OmegaConf.load(cfg_path)
    cfg.lm.use_flash_attn_2 = use_flash_attn
    cfg.mode = 'inference'
    
    if max_duration is not None:
        cfg.max_dur = max_duration
    
    # Force resolution of interpolations and patch problematic values
    cfg = _patch_config_interpolations(cfg)
    
    return cfg, model_path


def _patch_config_interpolations(cfg):
    """Patch any unresolved OmegaConf interpolations directly"""
    
    def patch_value(value):
        """Convert interpolated strings to actual values"""
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            # Extract the interpolation content
            interpolation = value[2:-1]  # Remove ${ and }
            
            if interpolation.startswith('eval:'):
                # Handle eval expressions like "eval:10*25+2"
                expr = interpolation[5:]  # Remove 'eval:'
                try:
                    result = eval(expr)
                    print(f"[DEBUG] Resolved {value} -> {result}")
                    return result
                except Exception as e:
                    print(f"[DEBUG] Failed to eval {expr}: {e}")
                    return 300  # Default fallback
            
            # Add other interpolation types as needed
            print(f"[DEBUG] Unknown interpolation type: {value}")
            return value
        
        return value
    
    def recursively_patch(obj):
        """Recursively patch all values in the config"""
        if isinstance(obj, dict):
            return {k: recursively_patch(v) for k, v in obj.items()}
        elif hasattr(obj, '_content'):  # OmegaConf DictConfig
            for key in obj:
                try:
                    obj[key] = recursively_patch(obj[key])
                except:
                    pass
            return obj
        elif isinstance(obj, (list, tuple)):
            return [recursively_patch(item) for item in obj]
        else:
            return patch_value(obj)
    
    return recursively_patch(cfg)


def load_audio_tokenizer(cfg):
    """Load audio tokenizer using the same logic as generate.py"""
    audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
    audio_tokenizer = audio_tokenizer.eval().cuda()
    return audio_tokenizer


def load_separate_tokenizer(cfg):
    """Load separate tokenizer if available, using generate.py logic"""
    if "audio_tokenizer_checkpoint_sep" in cfg.keys():
        seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
        seperate_tokenizer = seperate_tokenizer.eval().cuda()
        return seperate_tokenizer
    return None


def load_main_model(cfg, model_path, max_duration, seperate_tokenizer=None):
    """Load main model using the same logic as generate.py"""
    audiolm = builders.get_lm_model(cfg)
    checkpoint = torch.load(model_path, map_location='cpu')
    audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm = audiolm.eval()
    audiolm = audiolm.cuda().to(torch.float16)

    model = CodecLM(
        name="tmp",
        lm=audiolm,
        audiotokenizer=None,
        max_duration=max_duration,
        seperate_tokenizer=seperate_tokenizer,
    )
    
    return model


def set_generation_params(model, max_duration, temperature=0.9, cfg_coef=1.5, top_k=50, top_p=0.0):
    """Set generation parameters using generate.py defaults"""
    record_tokens = True
    record_window = 50
    
    model.set_generation_params(
        duration=max_duration, 
        extend_stride=5, 
        temperature=temperature, 
        cfg_coef=cfg_coef,
        top_k=top_k, 
        top_p=top_p, 
        record_tokens=record_tokens, 
        record_window=record_window
    )


def process_prompt_audio_item(item, separator, audio_tokenizer):
    """Process an item with prompt audio using exact logic from generate.py lines 111-144"""
    assert os.path.exists(item['prompt_audio_path']), f"prompt_audio_path {item['prompt_audio_path']} not found"
    assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
    
    with torch.no_grad():
        pmt_wav, vocal_wav, bgm_wav = separator.run(item['prompt_audio_path'])
    
    item['raw_pmt_wav'] = pmt_wav
    item['raw_vocal_wav'] = vocal_wav
    item['raw_bgm_wav'] = bgm_wav
    
    if pmt_wav.dim() == 2:
        pmt_wav = pmt_wav[None]
    if pmt_wav.dim() != 3:
        raise ValueError("Melody wavs should have a shape [B, C, T].")
    pmt_wav = list(pmt_wav)
    
    if vocal_wav.dim() == 2:
        vocal_wav = vocal_wav[None]
    if vocal_wav.dim() != 3:
        raise ValueError("Vocal wavs should have a shape [B, C, T].")
    vocal_wav = list(vocal_wav)
    
    if bgm_wav.dim() == 2:
        bgm_wav = bgm_wav[None]
    if bgm_wav.dim() != 3:
        raise ValueError("BGM wavs should have a shape [B, C, T].")
    bgm_wav = list(bgm_wav)
    
    if type(pmt_wav) == list:
        pmt_wav = torch.stack(pmt_wav, dim=0)
    if type(vocal_wav) == list:
        vocal_wav = torch.stack(vocal_wav, dim=0)
    if type(bgm_wav) == list:
        bgm_wav = torch.stack(bgm_wav, dim=0)
    
    pmt_wav = pmt_wav
    vocal_wav = vocal_wav
    bgm_wav = bgm_wav
    
    with torch.no_grad():
        pmt_wav, _ = audio_tokenizer.encode(pmt_wav.cuda())
    
    item['pmt_wav'] = pmt_wav
    item['vocal_wav'] = vocal_wav
    item['bgm_wav'] = bgm_wav
    item['melody_is_wav'] = False
    
    return item


def process_auto_prompt_item(item, auto_prompt):
    """Process an item with auto prompt using exact logic from generate.py lines 146-160"""
    assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
    
    if item["auto_prompt_audio_type"] == "Auto": 
        merge_prompt = [item for sublist in auto_prompt.values() for item in sublist]
        prompt_token = merge_prompt[np.random.randint(0, len(merge_prompt))]
    else:
        prompt_token = auto_prompt[item["auto_prompt_audio_type"]][np.random.randint(0, len(auto_prompt[item["auto_prompt_audio_type"]]))]
    
    item['pmt_wav'] = prompt_token[:,[0],:]
    item['vocal_wav'] = prompt_token[:,[1],:]
    item['bgm_wav'] = prompt_token[:,[2],:]
    item['melody_is_wav'] = False
    
    return item


def process_no_prompt_item(item):
    """Process an item without prompt using generate.py logic lines 156-160"""
    item['pmt_wav'] = None
    item['vocal_wav'] = None
    item['bgm_wav'] = None
    item['melody_is_wav'] = True
    return item


def apply_separate_tokenizer(item, seperate_tokenizer):
    """Apply separate tokenizer processing using generate.py logic lines 182-188"""
    if "prompt_audio_path" in item and seperate_tokenizer is not None:
        with torch.no_grad():
            vocal_wav, bgm_wav = seperate_tokenizer.encode(item['vocal_wav'].cuda(), item['bgm_wav'].cuda())
        item['vocal_wav'] = vocal_wav
        item['bgm_wav'] = bgm_wav
    return item


def generate_tokens(model, item):
    """Generate tokens using exact logic from generate.py lines 227-238"""
    lyric = item["gt_lyric"]
    descriptions = item["descriptions"] if "descriptions" in item else None
    pmt_wav = item['pmt_wav']
    vocal_wav = item['vocal_wav']
    bgm_wav = item['bgm_wav']
    melody_is_wav = item['melody_is_wav']

    generate_inp = {
        'lyrics': [lyric.replace("  ", " ")],
        'descriptions': [descriptions],
        'melody_wavs': pmt_wav,
        'vocal_wavs': vocal_wav,
        'bgm_wavs': bgm_wav,
        'melody_is_wav': melody_is_wav,
    }
    
    start_time = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            tokens = model.generate(**generate_inp, return_tokens=True)
    mid_time = time.time()
    
    return tokens, start_time, mid_time


def generate_audio_from_tokens(model, tokens, item, gen_type):
    """Generate audio from tokens using exact logic from generate.py lines 241-271"""
    with torch.no_grad():
        if 'raw_pmt_wav' in item:
            if gen_type == 'separate':
                wav_seperate = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='mixed')
                wav_vocal = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='vocal')
                wav_bgm = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='bgm')
                return wav_seperate, wav_vocal, wav_bgm
            elif gen_type == 'mixed':
                wav_seperate = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type=gen_type)
                return wav_seperate, None, None
            else:
                wav_seperate = model.generate_audio(tokens, chunked=True, gen_type=gen_type)
                return wav_seperate, None, None
        else:
            if gen_type == 'separate':
                wav_vocal = model.generate_audio(tokens, chunked=True, gen_type='vocal')
                wav_bgm = model.generate_audio(tokens, chunked=True, gen_type='bgm')
                wav_seperate = model.generate_audio(tokens, chunked=True, gen_type='mixed')
                return wav_seperate, wav_vocal, wav_bgm
            else:
                wav_seperate = model.generate_audio(tokens, chunked=True, gen_type=gen_type)
                return wav_seperate, None, None


def save_generated_audio(wav_seperate, wav_vocal, wav_bgm, target_wav_name, gen_type, sample_rate):
    """Save generated audio using exact logic from generate.py lines 266-271"""
    if gen_type == 'separate':
        torchaudio.save(target_wav_name.replace('.flac', '_vocal.flac'), wav_vocal[0].cpu().float(), sample_rate)
        torchaudio.save(target_wav_name.replace('.flac', '_bgm.flac'), wav_bgm[0].cpu().float(), sample_rate)
        torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), sample_rate)
        return {
            "main_audio": target_wav_name,
            "vocal_audio": target_wav_name.replace('.flac', '_vocal.flac'),
            "bgm_audio": target_wav_name.replace('.flac', '_bgm.flac')
        }
    else:
        torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), sample_rate)
        return {"main_audio": target_wav_name}


def cleanup_item_memory(item):
    """Clean up item memory using generate.py logic"""
    # Clean up raw audio data
    if 'raw_pmt_wav' in item:
        del item['raw_pmt_wav']
    if 'raw_vocal_wav' in item:
        del item['raw_vocal_wav']
    if 'raw_bgm_wav' in item:
        del item['raw_bgm_wav']
    
    # Clean up processed data
    if 'pmt_wav' in item:
        del item['pmt_wav']
    if 'vocal_wav' in item:
        del item['vocal_wav']
    if 'bgm_wav' in item:
        del item['bgm_wav']
    if 'melody_is_wav' in item:
        del item['melody_is_wav']


def create_output_directories(save_dir):
    """Create output directories using generate.py logic lines 213-215"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/audios", exist_ok=True)
    os.makedirs(save_dir + "/jsonl", exist_ok=True)


def cleanup_models_and_cache(audio_tokenizer=None, separator=None, seperate_tokenizer=None):
    """Clean up models and cache using generate.py logic lines 169-172"""
    if audio_tokenizer is not None:
        del audio_tokenizer
    if separator is not None:
        del separator
    if seperate_tokenizer is not None:
        del seperate_tokenizer
    
    torch.cuda.empty_cache()