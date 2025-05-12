# Indian Sign Language (ISL) Detection API

This application provides an API for detecting Indian Sign Language (ISL) signs from videos. It includes both a server-side component for processing videos and a client-side interface for recording and uploading videos.

## Features

- Upload and process pre-recorded videos
- Record videos directly in the browser and send for processing
- Detection of ISL signs using a pre-trained machine learning model
- JSON API for integration with other applications

## Setup and Installation

1. Clone the repository:

```
git clone <repository-url>
cd ISL_text
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
python app.py
```

The server will start on http://localhost:5000

## API Endpoints

### POST /api/process-video

Accepts a video file upload for sign language detection.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with a 'video' field containing the video file

**Response:**

```json
{
  "detected_signs": {
    "hello": 1,
    "thank you": 2,
    "help": 1
  },
  "total_signs": 4,
  "unique_signs": 3
}
```

### GET /api/live-capture

This is a demonstration endpoint that uses the server's webcam for sign language detection. Not recommended for production use.

**Request:**

- Method: GET

**Response:**
Same format as the `/api/process-video` endpoint.

## Client Usage

1. Open your browser and navigate to http://localhost:5000
2. Allow camera access when prompted
3. Click "Start Recording" to begin capturing sign language gestures
4. Perform the sign language gestures you want to detect
5. Click "Stop Recording" when finished
6. Click "Upload & Process" to send the video to the server for analysis
7. View the detected signs in the results section

## Technical Details

The application consists of several components:

- `app.py` - Flask web server and API endpoints
- `predictor.py` - Sign language detection model and video processing logic
- `static/index.html` - Client-side interface for recording and uploading videos

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- Flask
- MediaPipe
- Modern web browser with WebRTC support

## License

[MIT License](LICENSE)
