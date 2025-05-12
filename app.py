import os
import uuid
import traceback
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
from ISL_text.predictor import predict_isl_signs
from isl_nlp_pipeline.text_to_gloss.main import isl_pipeline
from isl_nlp_pipeline.gloss_to_text.gloss_to_english_copy import gloss_to_english
import requests

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


@app.route('/api/isl', methods=['POST'])
def process_isl():
    # Expect JSON with a "sentence" key
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    try:
        # Process the sentence using the pipeline
        isl_gloss = isl_pipeline(sentence)
        return jsonify({'isl_gloss': isl_gloss}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/english', methods=['POST'])
def process_english():
    # Expect JSON with a "gloss" key
    data = request.get_json()
    gloss = data.get('gloss', '')
    if not gloss:
        return jsonify({'error': 'No ISL gloss provided'}), 400

    try:
        # Process the gloss to generate English
        english_text = gloss_to_english(gloss)
        return jsonify({'english_text': english_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
                
                # Get ordered list of signs based on when they were detected
                # This will preserve temporal information for NLP processing
                ordered_signs = list(signs_dict.keys())
                
                # Clean up - delete the uploaded file after processing
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                print("signs_dict", signs_dict)
                print("ordered_signs", ordered_signs)

                # Return the results including the ordered list of signs
                return jsonify({
                    'detected_signs': signs_dict,
                    'ordered_signs': ordered_signs,
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

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """
    API endpoint to translate text to a target language using Google Translate API
    Expects JSON with:
    - text: The text to translate
    - target: The target language code (e.g., 'kn' for Kannada)
    """
    try:
        # Get the request data
        data = request.get_json()
        text = data.get('text', '')
        target_lang = data.get('target', 'kn')  # Default to Kannada if not specified
        
        if not text:
            return jsonify({'error': 'No text provided for translation'}), 400
        
        # Since we can't directly use the Node.js package in Python,
        # we'll use the free Google Translate API alternative
        url = "https://translate.googleapis.com/translate_a/single"
        
        params = {
            "client": "gtx",
            "sl": "en",  # Source language: English
            "tl": target_lang,  # Target language
            "dt": "t",  # Return translated text
            "q": text
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # Parse the response - it returns a nested array structure
            result = response.json()
            translated_text = ""
            
            # Extract the translated segments
            for segment in result[0]:
                if segment[0]:
                    translated_text += segment[0]
            
            return jsonify({
                'translatedText': translated_text,
                'sourceLang': 'en',
                'targetLang': target_lang
            }), 200
        else:
            return jsonify({
                'error': f'Translation API returned status code {response.status_code}'
            }), 500
            
    except Exception as e:
        error_msg = f"Translation error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({
            'error': error_msg
        }), 500

@app.route('/api/live-capture', methods=['GET'])
def live_capture():
    """
    API endpoint for demonstration purposes - 
    This will start the webcam on the server, which is generally not ideal
    for production environments.
    """
    try:
        # Use webcam (0) and show video for debugging
        signs_dict = predict_isl_signs(video_source=0, show_video=True)
        
        # Get ordered list of signs based on when they were detected
        ordered_signs = list(signs_dict.keys())
        
        return jsonify({
            'detected_signs': signs_dict,
            'ordered_signs': ordered_signs,
            'total_signs': sum(signs_dict.values()),
            'unique_signs': len(signs_dict)
        })
    except Exception as e:
        error_msg = f"Error during live capture: {str(e)}"
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