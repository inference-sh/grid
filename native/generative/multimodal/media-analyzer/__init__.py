"""
Media Analyzer App for inference.sh

This app analyzes images and audio files using OpenAI's multimodal AI models.
It supports both vision (images) and audio analysis in a single request.

Features:
- Image and audio file analysis
- Automatic metadata extraction (dimensions, duration, file size)
- Multi-file support
- Flexible model selection
"""

from .inference import App, AppInput, AppOutput, MediaMetadata

__all__ = ['App', 'AppInput', 'AppOutput', 'MediaMetadata']
