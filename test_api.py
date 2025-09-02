#!/usr/bin/env python3
"""
API Test Script for Object Detection API
Tests the detection endpoint with various configurations using test-image.jpg
"""

import requests
import base64
import json
import os
import sys
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:3011"
TEST_IMAGE_PATH = "test-image.jpg"
SECRET_KEY = "e2d4a6f453af4601b757f4f8ebfc6471"  # Replace with your actual secret key

def load_and_encode_image(image_path):
    """Load image and encode it to base64"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            encoded_string = base64.b64encode(image_data).decode('utf-8')
            print(f"✅ Image loaded and encoded: {len(encoded_string)} characters")
            return encoded_string
    except FileNotFoundError:
        print(f"❌ Error: Image file '{image_path}' not found")
        return None
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def test_health_check():
    """Test the health check endpoint"""
    print("\n🔍 Testing Health Check Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200 and "Endpoint reachable" in response.text:
            print("✅ Health check passed!")
            return True
        else:
            print("❌ Health check failed!")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_detection_endpoint(test_name, blur_faces, detect_objects, secret_key=SECRET_KEY):
    """Test the detection endpoint with specific configuration"""
    print(f"\n🔍 Testing: {test_name}")
    print(f"   blur-faces: {blur_faces}")
    print(f"   detect-objects: {detect_objects}")
    
    # Load and encode image
    encoded_image = load_and_encode_image(TEST_IMAGE_PATH)
    if not encoded_image:
        return False
    
    # Prepare request
    headers = {
        'Content-Type': 'application/json',
        'secret-key': secret_key,
        'blur-faces': blur_faces,
        'detect-objects': detect_objects
    }
    
    payload = {
        'image': encoded_image
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/detection", 
                               headers=headers, 
                               json=payload,
                               timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"Status: {result.get('status')}")
            
            # Show detection results
            if 'coordinates_data' in result:
                coords_data = result['coordinates_data']
                print(f"🎯 Objects detected:")
                total_objects = 0
                for class_name, detections in coords_data.items():
                    count = len(detections)
                    total_objects += count
                    print(f"   - {class_name}: {count} objects")
                    for i, detection in enumerate(detections):
                        coord = detection['coordinate']
                        conf = detection['confidence']
                        print(f"     [{i+1}] Box: [{coord[0]}, {coord[1]}, {coord[2]}, {coord[3]}], Confidence: {conf:.3f}")
                print(f"📊 Total objects detected: {total_objects}")
            
            # Show blur result
            if 'blured_image' in result:
                blur_result = result['blured_image']
                if blur_result:
                    print(f"🔒 Blurred image returned: {len(blur_result)} characters")
                else:
                    print(f"🔒 No blurring applied (no suitable faces found)")
            
            # Show system message
            if 'system_message' in result:
                print(f"💬 System message: {result['system_message']}")
            
            return True
            
        elif response.status_code == 401:
            print(f"❌ Unauthorized: {response.json()}")
            return False
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout (30s)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON response: {response.text}")
        return False

def save_response_to_file(test_name, response_data):
    """Save response data to a file for inspection"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_response_{test_name.lower().replace(' ', '_')}_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(response_data, f, indent=2)
        print(f"📁 Response saved to: {filename}")
    except Exception as e:
        print(f"❌ Could not save response: {e}")

def main():
    """Main test function"""
    print("🚀 Starting API Tests")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print("=" * 50)
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Test image '{TEST_IMAGE_PATH}' not found!")
        print("Please make sure test-image.jpg is in the same directory as this script.")
        sys.exit(1)
    
    # Test 1: Health Check
    health_ok = test_health_check()
    if not health_ok:
        print("❌ Health check failed. Is the Docker container running on port 3011?")
        print("Try: docker run -p 3011:8080 test:timelapse")
        sys.exit(1)
    
    # Test 2: Detection only
    test_detection_endpoint("Object Detection Only", "false", "true")
    
    # Test 3: Face blur only
    test_detection_endpoint("Face Blur Only", "true", "false")
    
    # Test 4: Both detection and blur
    test_detection_endpoint("Detection + Face Blur", "true", "true")
    
    # Test 5: Invalid headers (both false)
    test_detection_endpoint("Invalid Headers (both false)", "false", "false")
    
    # Test 6: Missing headers
    test_detection_endpoint("Missing Headers", "", "")
    
    # Test 7: Wrong secret key
    test_detection_endpoint("Wrong Secret Key", "true", "true", "wrong-key")
    
    print("\n" + "=" * 50)
    print("🏁 All tests completed!")
    print("\n💡 Tips:")
    print("- Update SECRET_KEY variable if you're using authentication")
    print("- Check Docker logs if tests fail: docker logs <container_id>")
    print("- Verify container is running: docker ps")

if __name__ == "__main__":
    main()