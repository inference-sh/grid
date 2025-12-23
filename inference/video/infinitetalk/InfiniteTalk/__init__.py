# Import src module and make it available globally
from . import src
import sys

# Make src available globally so 'from src.vram_management import' works
sys.modules['src'] = src