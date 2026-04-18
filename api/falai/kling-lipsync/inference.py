"""
Kling Lip-Sync

Lip-sync a source video to audio or text input using Kling's lipsync model.
Supports audio-to-video and text-to-video modes with configurable voice settings.
"""

from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File, OutputMeta, VideoMeta
from pydantic import Field, model_validator
from typing import Optional
from enum import Enum
import logging

from .fal_helper import setup_fal_client, run_fal_model, download_video

logging.getLogger("httpx").setLevel(logging.WARNING)


class ModeEnum(str, Enum):
    """Lip-sync mode — audio drives from an audio file, text uses TTS."""
    audio = "audio"
    text = "text"


class VoiceIdEnum(str, Enum):
    """Available voice IDs for text-to-video mode."""
    genshin_vindi2 = "genshin_vindi2"
    zhinen_xuesheng = "zhinen_xuesheng"
    AOT = "AOT"
    ai_shatang = "ai_shatang"
    genshin_klee2 = "genshin_klee2"
    genshin_kirara = "genshin_kirara"
    ai_kaiya = "ai_kaiya"
    oversea_male1 = "oversea_male1"
    ai_chenjiahao_712 = "ai_chenjiahao_712"
    girlfriend_4_speech02 = "girlfriend_4_speech02"
    chat1_female_new_3 = "chat1_female_new-3"
    chat_0407_5_1 = "chat_0407_5-1"
    cartoon_boy_07 = "cartoon-boy-07"
    uk_boy1 = "uk_boy1"
    cartoon_girl_01 = "cartoon-girl-01"
    PeppaPig_platform = "PeppaPig_platform"
    ai_huangzhong_712 = "ai_huangzhong_712"
    ai_huangyaoshi_712 = "ai_huangyaoshi_712"
    ai_laoguowang_712 = "ai_laoguowang_712"
    chengshu_jiejie = "chengshu_jiejie"
    you_pingjing = "you_pingjing"
    calm_story1 = "calm_story1"
    uk_man2 = "uk_man2"
    laopopo_speech02 = "laopopo_speech02"
    heainainai_speech02 = "heainainai_speech02"
    reader_en_m_v1 = "reader_en_m-v1"
    commercial_lady_en_f_v1 = "commercial_lady_en_f-v1"
    tiyuxi_xuedi = "tiyuxi_xuedi"
    tiexin_nanyou = "tiexin_nanyou"
    girlfriend_1_speech02 = "girlfriend_1_speech02"
    girlfriend_2_speech02 = "girlfriend_2_speech02"
    zhuxi_speech02 = "zhuxi_speech02"
    uk_oldman3 = "uk_oldman3"
    dongbeilaotie_speech02 = "dongbeilaotie_speech02"
    chongqingxiaohuo_speech02 = "chongqingxiaohuo_speech02"
    chuanmeizi_speech02 = "chuanmeizi_speech02"
    chaoshandashu_speech02 = "chaoshandashu_speech02"
    ai_taiwan_man2_speech02 = "ai_taiwan_man2_speech02"
    xianzhanggui_speech02 = "xianzhanggui_speech02"
    tianjinjiejie_speech02 = "tianjinjiejie_speech02"
    diyinnansang_DB_CN_M_04_v2 = "diyinnansang_DB_CN_M_04-v2"
    yizhipiannan_v1 = "yizhipiannan-v1"
    guanxiaofang_v2 = "guanxiaofang-v2"
    tianmeixuemei_v1 = "tianmeixuemei-v1"
    daopianyansang_v1 = "daopianyansang-v1"
    mengwa_v1 = "mengwa-v1"


class VoiceLanguageEnum(str, Enum):
    """Language for TTS voice."""
    zh = "zh"
    en = "en"


ENDPOINT_BY_MODE = {
    ModeEnum.audio: "fal-ai/kling-video/lipsync/audio-to-video",
    ModeEnum.text: "fal-ai/kling-video/lipsync/text-to-video",
}


class AppInput(BaseAppInput):
    """Input schema for Kling Lip-Sync."""

    video: File = Field(
        description="Source video to lip-sync. The model will modify lip movements to match the audio or text.",
    )
    mode: ModeEnum = Field(
        default=ModeEnum.audio,
        description="Lip-sync mode. Audio mode uses an audio file to drive lip movements, text mode uses TTS.",
    )
    audio: Optional[File] = Field(
        default=None,
        description="Driving audio file for audio mode. Must be 2-120 seconds long.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Text input for text mode. Maximum 120 characters.",
    )
    voice_id: Optional[VoiceIdEnum] = Field(
        default=None,
        description="Voice ID for text mode TTS. Required when mode is text.",
    )
    voice_language: Optional[VoiceLanguageEnum] = Field(
        default=None,
        description="Language for TTS voice (zh or en).",
    )
    voice_speed: Optional[float] = Field(
        default=1.0,
        description="Speech rate for TTS voice. 1.0 is normal speed.",
    )

    @model_validator(mode="after")
    def validate_mode_inputs(self):
        if self.mode == ModeEnum.audio and self.audio is None:
            raise ValueError("Audio file is required when mode is 'audio'.")
        if self.mode == ModeEnum.text:
            if self.text is None:
                raise ValueError("Text is required when mode is 'text'.")
            if self.voice_id is None:
                raise ValueError("voice_id is required when mode is 'text'.")
        return self


class AppOutput(BaseAppOutput):
    """Output schema for Kling Lip-Sync."""

    video: File = Field(description="The lip-synced video.")


class App(BaseApp):
    """Kling Lip-Sync application."""

    async def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Kling Lip-Sync initialized")

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            setup_fal_client()

            model_id = ENDPOINT_BY_MODE[input_data.mode]
            self.logger.info(
                f"Generating lip-sync video ({input_data.mode.value} mode), endpoint={model_id}"
            )

            request_data = {
                "video_url": input_data.video.uri,
            }

            if input_data.mode == ModeEnum.audio:
                request_data["audio_url"] = input_data.audio.uri
            else:
                request_data["text"] = input_data.text
                request_data["voice_id"] = input_data.voice_id.value
                if input_data.voice_language is not None:
                    request_data["voice_language"] = input_data.voice_language.value
                if input_data.voice_speed is not None:
                    request_data["voice_speed"] = input_data.voice_speed

            result = run_fal_model(model_id, request_data, self.logger)

            video_url = result["video"]["url"]
            video_path = download_video(video_url, self.logger)

            output_meta = OutputMeta(
                outputs=[
                    VideoMeta(
                        extra={
                            "mode": input_data.mode.value,
                        },
                    )
                ]
            )

            return AppOutput(
                video=File(path=video_path),
                output_meta=output_meta,
            )

        except Exception as e:
            self.logger.error(f"Error during lip-sync generation: {e}")
            raise RuntimeError(f"Lip-sync generation failed: {str(e)}")
