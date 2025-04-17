from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from deepfake_detector import DeepfakeAudioDetector
import tempfile

app = Flask(__name__)

# Initialize detector
detector = DeepfakeAudioDetector()

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()  # Use system temp directory
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Deepfake audio detector is running"})

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Run prediction
        result = detector.predict(filepath)
        
        # Add file info to result
        result["filename"] = filename
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    # Check if JSON data was provided
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get file paths from request
    data = request.get_json()
    if 'file_paths' not in data or not isinstance(data['file_paths'], list):
        return jsonify({"error": "JSON must include 'file_paths' list"}), 400
    
    file_paths = data['file_paths']
    
    # Check if paths exist
    valid_paths = [path for path in file_paths if os.path.exists(path)]
    invalid_paths = [path for path in file_paths if not os.path.exists(path)]
    
    if not valid_paths:
        return jsonify({"error": "None of the provided file paths exist"}), 400
    
    # Run predictions
    results = detector.batch_predict(valid_paths)
    
    # Add invalid paths to results
    for path in invalid_paths:
        results[path] = {"error": "File not found"}
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)