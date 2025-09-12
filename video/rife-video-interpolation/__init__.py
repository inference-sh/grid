# Make RIFE modules globally available for imports
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add ECCV2022-RIFE to path for model imports
rife_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ECCV2022-RIFE')
if rife_path not in sys.path:
    sys.path.append(rife_path)

from .inference import App, AppInput, AppOutput
