import os
import uuid
import traceback
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
from predictor import predict_isl_signs

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Increased to 100 MB max upload size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/process-video', methods=['POST'])
def process_video():
    try:
        # Check if the post request has the file part
        if 'video' not in request.files:
            return jsonify({
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        
        # If user did not select a file
        if file.filename == '':
            return jsonify({
                'error': 'No video file selected'
            }), 400
        
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(filepath)
            
            try:
                # Process the video with our predictor
                print(f"Processing video: {filepath}")
                signs_dict = predict_isl_signs(video_source=filepath, show_video=False)
                print(f"Processing complete. Detected signs: {signs_dict}")
                
                # Clean up - delete the uploaded file after processing
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                # Return the results
                return jsonify({
                    'detected_signs': signs_dict,
                    'total_signs': sum(signs_dict.values()),
                    'unique_signs': len(signs_dict)
                })
                
            except Exception as e:
                # Log the full error with traceback
                error_msg = f"Error processing video: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                
                # Clean up in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                return jsonify({
                    'error': error_msg
                }), 500
        
        return jsonify({
            'error': 'Invalid file type. Allowed types are: mp4, avi, mov, webm'
        }), 400
    
    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({
            'error': error_msg
        }), 500


@app.route('/', methods=['GET'])
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    print("Starting ISL Detection API server...")
    app.run(host='0.0.0.0', port=5000, debug=True) 