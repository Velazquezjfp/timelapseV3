import cv2
import base64
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN

def face_detect(person_coordinates):
    # Initialize MTCNN for face detection with optimized parameters
    print('starting face detect module')
    mtcnn = MTCNN(
        keep_all=True, 
        min_face_size=20,  # Minimum face size to detect
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load the original image
    image = cv2.imread('./image.jpg')
    if image is None:
        raise ValueError("Could not read the image file")

    # Flag to track if any faces were detected
    faces_detected = False

    # Process each person's coordinates
    for (px, py, pw, ph) in person_coordinates:
        # Define the subframe (region of interest) in the image
        subframe = image[py:py+ph, px:px+pw]

        # Skip very small subframes
        if subframe.size == 0 or pw < 50 or ph < 50:
            continue

        try:
            # Convert subframe to PIL Image for MTCNN
            img_pil = Image.fromarray(cv2.cvtColor(subframe, cv2.COLOR_BGR2RGB))
            
            # Detect faces in the subframe
            boxes, probs = mtcnn.detect(img_pil)

            # Check if faces are detected with a reasonable confidence
            if boxes is not None and len(boxes) > 0 and any(prob > 0.7 for prob in probs):
                faces_detected = True
                
                # Blur each detected face within the subframe
                for (fx, fy, fw, fh), prob in zip(boxes, probs):
                    if prob > 0.7:
                        # Calculate absolute coordinates for each face in the original image
                        face_x = px + int(fx)
                        face_y = py + int(fy)
                        face_w = int(fw - fx)
                        face_h = int(fh - fy)

                        # Extract and blur the face area in the original image
                        face_area = image[face_y:face_y+face_h, face_x:face_x+face_w]
                        blurred_face = cv2.GaussianBlur(face_area, (99, 99), 30)
                        image[face_y:face_y+face_h, face_x:face_x+face_w] = blurred_face

                        print(f"Blurred face at: x={face_x}, y={face_y}, w={face_w}, h={face_h}")

        except Exception as e:
            print(f'Issue with face subframe: {e}')

    # Only return base64 image if faces were detected
    if faces_detected:
        # Convert the image to base64
        _, buffer = cv2.imencode('.jpg', image)
        blured_image_base64 = base64.b64encode(buffer).decode('utf-8')
        return blured_image_base64
    
    # Return None if no faces were detected
    return None