import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import MODEL_PATH


def create_landmarker():
    """Initialize and return a MediaPipe HandLandmarker."""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, rgb_frame):
    """
    Run hand detection on an RGB frame.

    Returns
    -------
    mediapipe HandLandmarkerResult
    """
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    return landmarker.detect(mp_image)
