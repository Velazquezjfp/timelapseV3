# Object Detection & Privacy Blur API (v4)

A REST API for detecting vehicles, people, and construction equipment in images, with automatic head/face blurring for privacy protection.

## Features

- **Object Detection**: YOLOv7-E6 model detecting: `person`, `vehicle`, `construction_vehicle`, `bus`, `trailer`
- **Privacy Blurring**: MediaPipe Pose-based head detection with automatic blurring
- **Fallback System**: Top 25% blur for standing persons when head detection fails
- **Two Blur Modes**: `standard` (all persons) or `fast` (skip small detections)

---

## Environment Variables

Create a `.env` file in the project root:

```bash
APP_SECRET_KEY=your_secret_key_here
```

---

## Deployment

### Local Development

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd API-v3

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download yolov7_model (if not present)
gdown --id 1o1NzJzR0ps8w0J0LAidr5eQf1DfBvT9j -O yolov7_model.zip
unzip yolov7_model.zip
rm yolov7_model.zip

# 5. Create .env file
echo "APP_SECRET_KEY=your_secret_key_here" > .env

# 6. Run the app
python -c "from app import app; app.run(host='0.0.0.0', port=8080, debug=True)"
```

### Production (Docker)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd API-v3

# 2. Build the Docker image (automatically downloads yolov7_model)
docker build -t detection-api .

# 3. Run the container
docker run -d \
  -p 80:8080 \
  -e APP_SECRET_KEY=your_secret_key_here \
  --name detection-api \
  detection-api

# Your API is now available at: http://<your-server-ip>/detection
```

### Stop/Restart Container

```bash
# Stop
docker stop detection-api

# Remove
docker rm detection-api

# Rebuild and restart
docker build -t detection-api .
docker run -d -p 80:8080 -e APP_SECRET_KEY=your_secret_key_here --name detection-api detection-api
```

---

## API Reference

### Health Check

```
GET /
Response: "Endpoint reachable"
```

### Detection Endpoint

```
POST /detection
```

**Headers:**

| Header | Required | Values | Description |
|--------|----------|--------|-------------|
| `secret-key` | Yes | string | Authentication key |
| `blur-faces` | Yes | `true` / `false` | Enable head/face blurring |
| `detect-objects` | Yes | `true` / `false` | Return detection coordinates |
| `blur-mode` | No | `standard` / `fast` | Blur processing mode (default: `standard`) |

**Body:**

```json
{
  "image": "<base64_encoded_image>"
}
```

**Response:**

```json
{
  "status": "success",
  "coordinates_data": {
    "person": [{"coordinate": [x, y, w, h], "confidence": 0.85}, ...],
    "vehicle": [...],
    "construction_vehicle": [...]
  },
  "blured_image": "<base64_encoded_blurred_image_or_null>"
}
```

### Blur Modes

| Mode | Behavior |
|------|----------|
| `standard` | Process all detected persons |
| `fast` | Skip persons with height < 10% of image height |

---

## Project Structure

```
API-v3/
├── app.py                 # Flask API entry point
├── detector.py            # YOLOv7 object detection
├── face_module/           # Head detection & blurring (v4)
│   ├── __init__.py
│   ├── head_detector.py   # Main orchestrator
│   ├── pose_head.py       # MediaPipe Pose integration
│   └── blur_utils.py      # Blurring utilities
├── yolov7_model/          # YOLOv7 model files (downloaded)
├── requirements.txt
├── dockerfile
└── .env                   # Environment variables (create this)
```

---

## Testing

```bash
python test_api.py
```

Make sure to update the `SECRET_KEY` and `API_BASE_URL` in `test_api.py` to match your configuration.
