"""
Face Module v4 - Head Detection and Privacy Blurring

Replaces MTCNN-based face detection with MediaPipe Pose-based head detection.
Works better on low-resolution images and can detect heads even when
the person is facing away from the camera.

Usage:
    from face_module import blur_heads

    result = blur_heads(
        person_coordinates=[[x, y, w, h], ...],
        image_path='./image.jpg',
        original_size=(img_width, img_height),
        mode='standard'  # or 'fast'
    )

Modes:
    - 'standard': Process all person bounding boxes
    - 'fast': Skip subframes where person height < 10% of original image

Returns:
    Base64 encoded JPEG string with blurred heads, or None if no blurring applied.
"""

from .head_detector import blur_heads, get_processing_stats, should_process_person
from .pose_head import detect_head_in_subframe, get_pose_head_detector
from .blur_utils import (
    apply_gaussian_blur,
    apply_fallback_blur,
    encode_image_base64,
    calculate_head_bbox_from_landmarks
)

__version__ = '4.0.0'
__all__ = [
    'blur_heads',
    'get_processing_stats',
    'should_process_person',
    'detect_head_in_subframe',
    'get_pose_head_detector',
    'apply_gaussian_blur',
    'apply_fallback_blur',
    'encode_image_base64',
    'calculate_head_bbox_from_landmarks',
]
