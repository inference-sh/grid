# Import and make modules available for both relative and absolute imports
import sys
import os

# Regular imports
from . import configs, distributed, modules
from .first_last_frame2video import WanFLF2V
from .image2video import WanI2V
from .text2video import WanT2V
from .vace import WanVace, WanVaceMP

# Import utils to make wan.utils available
from . import utils

# Make wan module available as if imported from wan
current_module = sys.modules[__name__]
sys.modules['wan'] = current_module

# Now import multitalk after setting up the module paths
from .multitalk import InfiniteTalkPipeline
