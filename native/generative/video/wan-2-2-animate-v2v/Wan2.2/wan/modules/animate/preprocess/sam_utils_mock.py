# Mock SAM2 utils to avoid import errors when SAM2 is not needed
def build_sam2_video_predictor(model_cfg, checkpoint_path):
    """Mock function - SAM2 not available"""
    raise NotImplementedError("SAM2 is not available. Replacement mode is disabled.")
