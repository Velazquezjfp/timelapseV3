# Version 4.0.0 - Head Detection Module

## Overview

Replaced MTCNN-based face detection with MediaPipe Pose-based head detection for more robust privacy blurring.

## Changes Summary

### Dependencies Removed

| Package | Version | Purpose |
|---------|---------|---------|
| `facenet-pytorch` | 2.6.0 | MTCNN face detection (no longer needed) |

### Dependencies Added

| Package | Version | Purpose |
|---------|---------|---------|
| `mediapipe` | >=0.10.0 | Google's pose estimation for head detection |

### Files Removed

| File | Purpose |
|------|---------|
| `face_detect.py` | Old MTCNN-based face detection module |

### Files Added

| File | Purpose |
|------|---------|
| `face_module/__init__.py` | Module exports and documentation |
| `face_module/head_detector.py` | Main orchestrator with mode support |
| `face_module/pose_head.py` | MediaPipe Pose integration |
| `face_module/blur_utils.py` | Blurring utilities and fallback logic |

### Files Modified

| File | Changes |
|------|---------|
| `app.py` | Import from face_module, add blur-mode header, handle new detector return |
| `detector.py` | Return original image size along with detection results |
| `requirements.txt` | Remove facenet-pytorch, add mediapipe |

## New Features

### Blur Mode Header

New API header `blur-mode` controls processing behavior:
- `standard` (default): Process all person bounding boxes
- `fast`: Skip subframes where person height < 10% of original image

### Fallback Logic

When head detection fails or has low confidence (< 0.5):
- Check if person is standing (height > width * 1.5)
- If standing: blur top 25% of bounding box
- If not standing: skip (likely crouching or vehicle misdetection)

## Why MediaPipe Pose?

| Feature | MTCNN (old) | MediaPipe Pose (new) |
|---------|-------------|----------------------|
| GitHub Stars | - | 32.3k |
| Maintenance | Active | Active (Google) |
| Back of head | No | Yes (from body pose) |
| Side profile | Poor | Good |
| Low resolution | Poor | Good |
| Model download | Large | Lightweight |

## Testing

Run tests with:
```bash
python test_api.py
```

Expected improvements:
- Better detection on small person bounding boxes
- Blurring works even when person facing away
- Fallback ensures privacy protection when detection fails

## Rollback Instructions

If issues occur:
1. Restore `face_detect.py` from git
2. Revert `requirements.txt` changes
3. Revert `app.py` import line
4. Remove `face_module/` directory
