import cv2
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

# Add yolov7_model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov7_model'))

# Import YOLOv7 modules
from yolov7_model.models.experimental import attempt_load
from yolov7_model.utils.general import non_max_suppression, scale_coords, check_img_size
from yolov7_model.utils.datasets import letterbox
from yolov7_model.utils.torch_utils import select_device

class ConstructionVehicleDetector:
    """YOLOv7-E6 Construction Vehicle Detector for API"""
    
    def __init__(self, model_path="yolov7_model/best.pt", device='cpu', img_size=1280, conf_thres=0.25, iou_thres=0.45):
        """Initialize the detector"""
        try:
            self.device = select_device(device)
        except:
            self.device = torch.device('cpu')
        
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Class names mapping to original API format
        self.class_names = {
            0: 'bus',
            1: 'construction_vehicle', 
            2: 'person',
            3: 'trailer',
            4: 'vehicle'  # Changed from 'car' to 'vehicle' to match original API
        }
        
        # Load model
        self.model = None
        try:
            model_full_path = os.path.join(os.path.dirname(__file__), model_path)
            print(f"Loading model from: {model_full_path}")
            self.model = attempt_load(model_full_path, map_location=self.device)
            self.model.eval()
            
            # Check image size
            self.img_size = check_img_size(self.img_size, s=self.model.stride.max())
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Apply letterbox preprocessing (preserves aspect ratio with padding)
        img = letterbox(image, self.img_size, stride=32, auto=False)[0]
        
        # Convert BGR to RGB and normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # Normalize to 0-1
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # Add batch dimension
        
        return img
    
    def detect_objects(self, image):
        """Detect objects and return results in original API format"""
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                pred = self.model(img_tensor)[0]
                
                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                         classes=None, agnostic=False)
            
            # Process results
            result_dict = {}
            
            if pred[0] is not None and len(pred[0]):
                detections = pred[0].cpu().numpy()
                
                # Scale coordinates back to original image size
                scaled_detections = scale_coords(img_tensor.shape[2:], torch.tensor(detections[:, :4]), image.shape).round()
                
                # Process each detection
                for i, (*xyxy, conf, cls) in enumerate(detections):
                    cls_int = int(cls)
                    if cls_int in self.class_names:
                        class_name = self.class_names[cls_int]
                        
                        # Filter out low-confidence detections for the "person" class
                        if class_name == 'person' and conf < 0.40:
                            continue
                        
                        # Get scaled coordinates
                        x1, y1, x2, y2 = scaled_detections[i].int().tolist()
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Format as [x, y, w, h] to match original API
                        coordinate = [x1, y1, w, h]
                        
                        # Initialize class in result dict if not exists
                        if class_name not in result_dict:
                            result_dict[class_name] = []
                        
                        # Add detection
                        result_dict[class_name].append({
                            'coordinate': coordinate,
                            'confidence': float(conf)
                        })
            
            return result_dict
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return {}

# Global detector instance
_detector = None

def get_detector():
    """Get or create detector instance"""
    global _detector
    if _detector is None:
        _detector = ConstructionVehicleDetector()
    return _detector

def process_image_multi_detector(numpy_image):
    """Main function to maintain compatibility with original API.

    Returns:
        Tuple of (detection_results, original_size) where:
            - detection_results: dict with class names as keys
            - original_size: tuple (width, height) of original image
    """
    try:
        # Decode numpy array to image
        image_decode_bs64 = cv2.imdecode(numpy_image, cv2.IMREAD_COLOR)

        # Get original image dimensions (height, width, channels)
        img_h, img_w = image_decode_bs64.shape[:2]
        original_size = (img_w, img_h)

        # Save the image as jpg (keeping original functionality)
        cv2.imwrite('./image.jpg', image_decode_bs64)

        # Get detector and run inference
        detector = get_detector()
        result = detector.detect_objects(image_decode_bs64)

        print(f'Detection results: {result}')
        print(f'Original image size: {original_size}')
        return result, original_size

    except Exception as e:
        print(f"Error in process_image_multi_detector: {e}")
        return {}, (0, 0)