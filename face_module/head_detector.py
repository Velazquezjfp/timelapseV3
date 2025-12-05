"""
Main head detection orchestrator for privacy blurring.
Coordinates MediaPipe Pose detection with fallback logic.
"""

import cv2
from .pose_head import detect_head_in_subframe
from .blur_utils import apply_gaussian_blur, apply_fallback_blur, encode_image_base64


# Configuration constants
MIN_RELATIVE_HEIGHT = 0.10  # 10% of original image height for 'fast' mode
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to trust detection
MIN_SUBFRAME_SIZE = 30  # Minimum pixel size for subframe processing


def blur_heads(person_coordinates, image_path, original_size, mode='standard'):
    """
    Main function to blur heads in detected person regions.

    Args:
        person_coordinates: List of [x, y, w, h] bounding boxes from detector.py
        image_path: Path to the saved image file (e.g., './image.jpg')
        original_size: Tuple (width, height) of original image before processing
        mode: 'standard' (process all) or 'fast' (skip small subframes)

    Returns:
        Base64 encoded blurred image string, or None if no blurring applied
    """
    print(f'Starting head detection module (mode={mode})')

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    img_h, img_w = image.shape[:2]
    original_w, original_h = original_size

    # Track if any blurring was applied
    blur_applied = False

    # Process each person bounding box
    for idx, (px, py, pw, ph) in enumerate(person_coordinates):
        print(f"Processing person {idx + 1}/{len(person_coordinates)}: bbox=({px}, {py}, {pw}, {ph})")

        # Calculate relative height compared to original image
        relative_height = ph / original_h

        # Fast mode: skip small subframes
        if mode == 'fast' and relative_height < MIN_RELATIVE_HEIGHT:
            print(f"  Skipping (fast mode): relative_height={relative_height:.3f} < {MIN_RELATIVE_HEIGHT}")
            continue

        # Skip very small subframes (would fail anyway)
        if pw < MIN_SUBFRAME_SIZE or ph < MIN_SUBFRAME_SIZE:
            print(f"  Skipping: subframe too small ({pw}x{ph} < {MIN_SUBFRAME_SIZE})")
            continue

        # Extract subframe
        subframe = image[py:py+ph, px:px+pw]
        if subframe.size == 0:
            print(f"  Skipping: empty subframe")
            continue

        # Detect head using MediaPipe Pose
        head_bbox, confidence, is_valid = detect_head_in_subframe(
            subframe,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        if is_valid and head_bbox is not None:
            # Convert subframe-relative coordinates to image-absolute coordinates
            hx, hy, hw, hh = head_bbox
            abs_head_bbox = (px + hx, py + hy, hw, hh)

            # Apply blur to detected head
            if apply_gaussian_blur(image, abs_head_bbox):
                blur_applied = True
                print(f"  Blurred head at: x={px + hx}, y={py + hy}, w={hw}, h={hh} (confidence={confidence:.2f})")
            else:
                print(f"  Failed to apply blur to detected head")

        else:
            # No valid detection or low confidence - apply fallback
            print(f"  No valid head detection (confidence={confidence:.2f}), checking fallback...")

            # Use original person bbox for fallback
            person_bbox = (px, py, pw, ph)
            if apply_fallback_blur(image, person_bbox):
                blur_applied = True
            else:
                print(f"  Fallback not applied (person not standing)")

    # Return result
    if blur_applied:
        return encode_image_base64(image)
    else:
        print("No blurring applied to any person")
        return None


def should_process_person(person_bbox, original_size, mode):
    """
    Determine if a person bounding box should be processed.

    Args:
        person_bbox: Tuple (x, y, w, h)
        original_size: Tuple (width, height) of original image
        mode: 'standard' or 'fast'

    Returns:
        True if should process, False to skip
    """
    _, _, pw, ph = person_bbox
    original_w, original_h = original_size

    # Always skip very small subframes
    if pw < MIN_SUBFRAME_SIZE or ph < MIN_SUBFRAME_SIZE:
        return False

    # In fast mode, skip based on relative height
    if mode == 'fast':
        relative_height = ph / original_h
        return relative_height >= MIN_RELATIVE_HEIGHT

    return True


def get_processing_stats(person_coordinates, original_size, mode):
    """
    Get statistics about which person boxes will be processed.
    Useful for debugging and logging.

    Args:
        person_coordinates: List of [x, y, w, h] bounding boxes
        original_size: Tuple (width, height)
        mode: 'standard' or 'fast'

    Returns:
        Dictionary with processing statistics
    """
    total = len(person_coordinates)
    will_process = sum(
        1 for bbox in person_coordinates
        if should_process_person(bbox, original_size, mode)
    )
    will_skip = total - will_process

    return {
        'total_persons': total,
        'will_process': will_process,
        'will_skip': will_skip,
        'mode': mode
    }
