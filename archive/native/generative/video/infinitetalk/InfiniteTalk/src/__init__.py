# Make modules publicly available with both relative and absolute imports
from . import vram_management
from . import utils
from .vram_management import *

# Also make them available as if imported from src
import sys
import os
current_module = sys.modules[__name__]
sys.modules['src'] = current_module