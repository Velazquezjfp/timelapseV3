"""
MediaPipe Pose-based head detection module.
Uses body pose landmarks to locate head position, which works even when
the person is facing away from the camera.
"""

import cv2
import mediapipe as mp
from .blur_utils import calculate_head_bbox_from_landmarks


class PoseHeadDetector:
    """
    Head detector using MediaPipe Pose estimation.
    Extracts head region from body pose landmarks.
    """

    # MediaPipe Pose landmark indices for head-related points
    LANDMARK_NAMES = {
        0: 'nose',
        1: 'left_eye_inner',
        2: 'left_eye',
        3: 'left_eye_outer',
        4: 'right_eye_inner',
        5: 'right_eye',
        6: 'right_eye_outer',
        7: 'left_ear',
        8: 'right_ear',
    }

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the Pose Head Detector.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Process individual images, not video
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        print("PoseHeadDetector initialized with MediaPipe Pose")

    def detect_head(self, subframe):
        """
        Detect head region in a person subframe using pose estimation.

        Args:
            subframe: OpenCV image (BGR) of a cropped person region

        Returns:
            Tuple of (head_bbox, confidence) where:
                - head_bbox: (x, y, w, h) relative to subframe, or None
                - confidence: Average visibility of head landmarks (0-1)
        """
        if subframe is None or subframe.size == 0:
            return None, 0.0

        h, w = subframe.shape[:2]
        if h < 20 or w < 20:  # Too small for pose detection
            return None, 0.0

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(subframe, cv2.COLOR_BGR2RGB)

            # Run pose detection
            results = self.pose.process(rgb_image)

            if not results.pose_landmarks:
                return None, 0.0

            # Extract head landmarks
            landmarks = {}
            visibilities = []

            for idx, name in self.LANDMARK_NAMES.items():
                landmark = results.pose_landmarks.landmark[idx]
                # Convert normalized coordinates to pixel coordinates
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility

                landmarks[name] = (x, y, visibility)
                visibilities.append(visibility)

            # Calculate average confidence
            avg_confidence = sum(visibilities) / len(visibilities) if visibilities else 0.0

            # Calculate head bounding box from landmarks
            head_bbox = calculate_head_bbox_from_landmarks(
                landmarks,
                person_bbox=(0, 0, w, h),
                image_shape=subframe.shape
            )

            return head_bbox, avg_confidence

        except Exception as e:
            print(f"Error in pose detection: {e}")
            return None, 0.0

    def close(self):
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()


# Singleton instance for efficiency
_detector = None


def get_pose_head_detector():
    """Get or create singleton PoseHeadDetector instance."""
    global _detector
    if _detector is None:
        _detector = PoseHeadDetector()
    return _detector


def detect_head_in_subframe(subframe, confidence_threshold=0.5):
    """
    Convenience function to detect head in a subframe.

    Args:
        subframe: OpenCV image (BGR) of a cropped person region
        confidence_threshold: Minimum confidence to consider detection valid

    Returns:
        Tuple of (head_bbox, confidence, is_valid) where:
            - head_bbox: (x, y, w, h) relative to subframe, or None
            - confidence: Detection confidence (0-1)
            - is_valid: True if confidence >= threshold
    """
    detector = get_pose_head_detector()
    head_bbox, confidence = detector.detect_head(subframe)

    is_valid = head_bbox is not None and confidence >= confidence_threshold

    return head_bbox, confidence, is_valid
