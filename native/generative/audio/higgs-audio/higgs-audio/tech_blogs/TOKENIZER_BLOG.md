# Higgs Audio Tokenizer

In this work, we introduce a new discretized audio tokenizer that runs at just **25 frames per second** while keeping—or even improving—audio quality compared to tokenizers with twice the bitrate. Our model is the first to train on **24 kHz data** covering speech, music, and sound events in one unified system. It also uses a simple non-diffusion encoder/decoder for fast, batch inference.

![XCodec Architecture](../figures/higgs_audio_tokenizer_architecture.png)

## Basics of Audio Quantization

An audio signal sampled at $f_s$ Hz is first split into frames by an encoder with hop size $M$, giving a frame rate  $f_r = \frac{f_s}{M}\quad\text{(frames/s)}.$
Two common quantizers are:

- **Residual Vector Quantization (RVQ)**: $N_q$ cascaded layers with codebook size $N_{cb}$ each. When $N_{cb}=1$, it reduces to single-vector quantization.  
- **Finite Scalar Quantization (FSQ)**: A single layer ($N_q=1$) with codebook size $N_{cb}$.  

If every combination of codewords is a token, the vocabulary size is $N_{cb}^{N_q}$, and each token needs $N_q\log_2 N_{cb}$ bits. The overall bitrate (bits/s, BPS) is simply $f_r \times N_q \log_2 N_{cb}.$  
We aim to push this bitrate as low as possible without hurting audio fidelity.

## What Makes Ours Better

- **Low Frame Rate**: At 25 fps, our tokenizer halves the frame rate of many baselines when still maintaining high audio quality. 
- **Unified 24 kHz Training**: We mix speech, music, and sound-event clips in one model, capturing both semantic and acoustic details, hugely facilitating the training of audio language models.
- **Fast Inference**: By avoiding diffusion steps, our encoder/decoder processes batches quickly, making it practical for real-time or large-scale tasks.


## Data and Evaluation Metrics

We test on four subsets, available [here](https://huggingface.co/datasets/bosonai/AudioTokenBench):

- **Speech, Music, Sound Event**: Includes 1,000 clips for each category, with each clip lasting 10 seconds.   Clips are randomly sampled from [DAPS](https://ccrma.stanford.edu/~gautham/Site/daps.html) (Speech), [MUSDB](https://sigsep.github.io/datasets/musdb.html) (Music), and [AudioSet](https://research.google.com/audioset/index.html) (Sound Event).

- **Audiophile**: Contains 150 clips, each 30 seconds long, curated from eleven high-fidelity test discs. The clips feature both music and sound events, selected for audio quality evaluation.

We measure:

- **Acoustic Quality**: STFT distance between the original and reconstructed audio.  
- **Semantic Integrity**: Semantic preservation of the original audio using [SeedTTS](https://arxiv.org/abs/2406.02430)[15] dataset on English and Chinese. 
- **Aesthetics**: SOTA unified model-based quality assessment, [Meta Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)[8], for Content Enjoyment (CE), Content Usefulness (CU) .


We compare our tokenizer with a wide range of baselines, from tokenizers mainly built for better acoustic reconstruction and compression rate, to those focused on semantic integrity, and to tokenizers used in existing large audio language models. We also compare with tokenizers that are pretrained specifically on speech or on music.


The tables below summarize the tokenizers evaluated. As shown, our tokenizer achieves a well-rounded balance of efficiency, semantic fidelity, and acoustic quality.

### Accoustic Evaluation

We use the STFT metric here for simplicity. The baselines are ordered chronologically, grouped by whether semantic distillation (SD) is applied.Despite DAC’s top acoustic quality at 12× the bitrate, our tokenizer leads all other baselines.


| Tokenizer | 💬 | 🎵 | 🥁 | SD | $f_s$ | $f_r$ | BPS* (k) ↓ | Speech ↓ | Sound Event ↓ | Music ↓ | Audiophile ↓ |
|-----------|----|----|----|----|-------|-------|--------------------------|----------|----------------|--------|--------------|
| [Encodec](https://huggingface.co/facebook/encodec_24khz)[3] | ✓ | ✓ | ✓ |  | 24 | 75 | 24 | 1.96 | 2.65 | 2.52 | 2.30 |
| [DAC](https://huggingface.co/hance-ai/descript-audio-codec-24khz)[2] | ✓ | ✓ | ✓ |  | 24 | 75 | 24 | **1.13** | **1.45** | **1.34** | **1.62** |
| [SNAC-24k](https://huggingface.co/hubertsiuzdak/snac_24khz)[6] | ✓ |  |  |  | 24 | (12, 23, 47) | 0.98 | 1.92 | 2.69 | 2.54 | 2.52 |
| [SNAC-44.1k](https://huggingface.co/hubertsiuzdak/snac_44khz)[6] |  | ✓ | ✓ |  | 44.1 | (14, 29, 57, 115) | 2.6 | 1.83 | 2.25 | 2.05 | 2.00 |
| [WavTokenizer](https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/blob/main/wavtokenizer_medium_music_audio_320_24k_v2.ckpt)[7] |  | ✓ | ✓ |  | 24 | 75 | 0.9 | 1.93 | 2.44 | 2.17 | 2.15 |
| [WavTokenizer (Speech)](https://huggingface.co/novateur/WavTokenizer-large-speech-75token/tree/main)[7] | ✓ |  |  |  | 24 | 75 | 0.9 | 1.78 | 2.47 | 2.42 | 2.47 |
| [MuCodec](https://huggingface.co/haoheliu/audioldm_48k/tree/main)[11] |  | ✓ |  |  | 48 | 25 | 0.35 | 2.87 | 3.69 | 3.36 | 2.97 |
| [FlowDec-75m](https://github.com/facebookresearch/FlowDec?tab=readme-ov-file)[12] | ✓ | ✓ | ✓ |  | 48 | 75 | 7.5 | 1.73 | 2.14 | 2.01 | 2.03 |
| [FlowDec-25s](https://github.com/facebookresearch/FlowDec?tab=readme-ov-file)[12] | ✓ | ✓ | ✓ |  | 48 | 25 | 4 | 1.94 | 2.42 | 2.25 | 2.33 |
| [SpeechTokenizer](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg)[14] | ✓ |  |  | ✓ | 16 | 50 | 4 | 3.21 | 3.58 | 3.65 | 3.69 |
| [SemantiCodec](https://huggingface.co/haoheliu/SemantiCodec/tree/main/semanticodec_tokenrate_100)[5] | ✓ | ✓ | ✓ | ✓ | 16 | 50 | 1.4 | 3.05 | 3.28 | 3.24 | 3.18 |
| [Mimi](https://huggingface.co/docs/transformers/en/model_doc/mimi)[13] | ✓ |  |  | ✓ | 24 | 12.5 | 4.4 | 1.77 | 2.40 | 2.30 | 2.15 |
| [XCodec](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert_general.yaml)[1] | ✓ | ✓ | ✓ | ✓ | 16 | 50 | 4 | 2.95 | 3.16 | 3.00 | 3.03 |
| [CosyVoice 2](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)[13] | ✓ |  |  | ✓ | 16 | 25 | -** | 2.30 | 3.30 | 3.14 | 3.25 |
| [XCodec2](https://huggingface.co/HKUST-Audio/xcodec2/blob/main/ckpt/epoch%3D4-step%3D1400000.ckpt)[9] | ✓ |  |  | ✓ | 16 | 50 | 0.8 | 3.06 | 3.72 | 3.62 | 3.64 |
| [XY](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/tree/main)[10] | ✓ |  |  | ✓ | 24 | 12.5 | 1 | 1.89 | 2.51 | 2.40 | 2.26 |
| Ours | ✓ | ✓ | ✓ | ✓ | 24 | 25 | 2 | **1.62** | **2.03** | **1.85** | **1.80** |



\* Bits-per-second is calculated according to the checkpoint the author provided.

\*\* CosyVoice 2 uses the continuous feature as the conditioning, we include it for completeness.


### Semantic Evaluation
Here we only compare with tokenizers that are trained with semantic distillation.
[SeedTTS](https://github.com/BytedanceSpeech/seed-tts-eval) is a dataset includes prompt/target audio and texts. We reconstructed the target audio, and use the word error rate (WER) and speaker similarity (SIM) metrics to evaluate the semantic integrity. SIM is calculated by the similarity between the prompt audio and reconstructed targeted audio with [WavLM-large](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view) as the embedding model. 

The following table shows that our tokenizer achieves comparable performance to tokenizers that 2.2x the bitrate of our model.

| Model | BPS (k) | en WER ↓ | en SIM ↑ | zh WER ↓ | zh SIM ↑ |
|------------------|---------|------------|------------|------------|------------|
| [SpeechTokenizer](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg) | 4 | 2.82 | 0.63 | 2.04 | 0.65 |
| [SemantiCodec](https://huggingface.co/haoheliu/SemantiCodec/tree/main/semanticodec_tokenrate_100) | 1.4 | 3.46 | 0.56 | 2.18 | 0.60 |
| [Mimi](https://huggingface.co/docs/transformers/en/model_doc/mimi) | 4.4 | **2.35** | **0.70** | **1.48** | **0.72** |
| [XCodec](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert_general.yaml) | 4.0 | 2.68 | 0.63 | 1.66 | 0.66 |
| [CosyVoice 2](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | - | 3.17 | 0.65 | 2.11 | 0.70 |
| [XCodec2](https://huggingface.co/HKUST-Audio/xcodec2/blob/main/ckpt/epoch%3D4-step%3D1400000.ckpt) | 0.8 | 2.74 | 0.62 | 1.91 | 0.67 |
| [XY-MOSS-TTSD](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/tree/main) | 1.0 | 2.72 | 0.61 | 1.58 | 0.67 |
| Ours | 2.0 | 2.52 | 0.67 | **1.48** | 0.71 |



### Audiobox Aesthetics Evaluation

This model based evaluation[8] further demonstrates the superiority of our tokenizer. CU is the Content Usefulness and CE is the Content Enjoyment. Each term is rated on a scale of 1-10. Notably, our tokenizer performs best on the Audiophile set—demonstrating a clear advantage when the original audio quality is high.


| Model | BPS (k) | Music CE ↑ | Music CU ↑ | Sound Event CE ↑ | Sound Event CU ↑ | Speech CE ↑ | Speech CU ↑ | Audiophile CE ↑ | Audiophile CU ↑ |
|------------------|---------|--------------|--------------|--------------------|--------------------|---------------|---------------|--------------------|--------------------|
| Origin | - | 6.20 | 7.10 | 4.47 | 5.64 | 5.03 | 4.87 | 7.17 | 7.65 |
| [SpeechTokenizer](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg) | 4.0 | 3.55 | 5.22 | 3.03 | 4.50 | 4.68 | 4.58 | 3.59 | 5.07 |
| [SemantiCodec](https://huggingface.co/haoheliu/SemantiCodec/tree/main/semanticodec_tokenrate_100) | 1.4 | 6.01 | 6.83 | 4.22 | 5.30 | 4.28 | 4.12 | 6.97 | 7.43 |
| [Mimi](https://huggingface.co/docs/transformers/en/model_doc/mimi) | 4.4 | 6.01 | 6.83 | 4.26 | 5.35 | 4.87 | 4.72 | 6.80 | 7.29 |
| [XCodec](https://huggingface.co/ZhenYe234/xcodec/blob/main/config_hubert_general.yaml) | 4.0 | **6.30** | **7.10** | **4.43** | 5.45 | **4.96** | **4.79** | 7.06 | 7.49 |
| [CosyVoice 2](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | - | 5.21 | 6.14 | 4.08 | 4.73 | **4.91** | **4.75** | 5.97 | 6.56 |
| [XCodec2](https://huggingface.co/HKUST-Audio/xcodec2/blob/main/ckpt/epoch%3D4-step%3D1400000.ckpt) | 0.8 | 4.38 | 5.66 | 3.43 | 4.63 | **4.93** | **4.78** | 4.56 | 5.46 |
| [XY-MOSS-TTSD](https://huggingface.co/fnlp/XY_Tokenizer_TTSD_V0/tree/main) | 1.0 | 5.77 | 6.80 | 4.23 | 5.34 | 4.88 | 4.72 | 6.95 | 7.48 |
| Ours | 2.0 | **6.35** | **7.15** | **4.47** | **5.51** | 4.90 | 4.70 | **7.21** | **7.66** |



Note that since some tokenizers are trained on 16 kHz data, we upsample their audio outputs to 24 kHz before computing metrics. Different upsampling methods may cause slight variations (e.g., 4.36 vs. 4.43 for XCodec Sound Event CE). We report the best results we could obtain and highlight any results within 0.05 of the best one. 







<!-- xcodec [1]
dac [2]
encodec [3]
moshi [4]
semanticodec [5]
snac [6]
wavtokenizer [7]
xcodec [8]
xcodec2 [9]
xy-tokenizer [10]
mucodec [11]
flowdec [12]
cosyvoice2 [13]
speechtokenizer [14] -->




## Reference 
[1] [Ye, Zhen, et al. "Codec does matter: Exploring the semantic shortcoming of codec for audio language model." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 24. 2025.](https://arxiv.org/abs/2408.17175)

[2] [Kumar, Rithesh, et al. "High-fidelity audio compression with improved rvqgan." Advances in Neural Information Processing Systems 36 (2023): 27980-27993.](https://dl.acm.org/doi/10.5555/3666122.3667336)

[3] [Défossez, Alexandre, et al. "High fidelity neural audio compression." arXiv preprint arXiv:2210.13438 (2022).](https://arxiv.org/abs/2210.13438)

[4] [Défossez, Alexandre, et al. "Moshi: a speech-text foundation model for real-time dialogue." arXiv preprint arXiv:2410.00037 (2024).](https://arxiv.org/abs/2410.00037)

[5] [Liu, Haohe, et al. "Semanticodec: An ultra low bitrate semantic audio codec for general sound." IEEE Journal of Selected Topics in Signal Processing (2024).](https://ieeexplore.ieee.org/document/10768970)

[6] [Siuzdak, Hubert, Florian Grötschla, and Luca A. Lanzendörfer. "Snac: Multi-scale neural audio codec." arXiv preprint arXiv:2410.14411 (2024).](https://arxiv.org/abs/2410.14411)

[7] [Ji, Shengpeng, et al. "Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling." arXiv preprint arXiv:2408.16532 (2024).](https://arxiv.org/abs/2408.16532)

[8] [Tjandra, Andros, et al. "Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound." arXiv preprint arXiv:2502.05139 (2025).](https://arxiv.org/abs/2502.05139)

[9] [Ye, Zhen, et al. "Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis." arXiv preprint arXiv:2502.04128 (2025).](https://arxiv.org/abs/2502.04128)

[10] [Gong, Yitian, et al. "XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs." arXiv preprint arXiv:2506.23325 (2025).](https://arxiv.org/abs/2506.23325)

[11] [Xu, Yaoxun, et al. "MuCodec: Ultra Low-Bitrate Music Codec." arXiv preprint arXiv:2409.13216 (2024).](https://arxiv.org/abs/2409.13216)

[12] [Welker, Simon, et al. "FlowDec: A flow-based full-band general audio codec with high perceptual quality." arXiv preprint arXiv:2503.01485 (2025).](https://arxiv.org/abs/2503.01485)

[13] [Du, Zhihao, et al. "Cosyvoice 2: Scalable streaming speech synthesis with large language models." arXiv preprint arXiv:2412.10117 (2024).](https://arxiv.org/abs/2412.10117)

[14] [Zhang, Xin, et al. "Speechtokenizer: Unified speech tokenizer for speech large language models." arXiv preprint arXiv:2308.16692 (2023).](https://arxiv.org/abs/2308.16692)

[15] [Anastassiou, Philip, et al. "Seed-tts: A family of high-quality versatile speech generation models." arXiv preprint arXiv:2406.02430 (2024).](https://arxiv.org/abs/2406.02430)
