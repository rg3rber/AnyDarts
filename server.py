import os
import time
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
import ssl

app = Flask(__name__)
import logging
import requests
import os
from DartsPrediction import score

app = flask.Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['POST'])
def handle_image_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Secure filename and save
        filename = secure_filename(file.filename)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        full_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
        file.save(full_path)
        
        # Attempt to score the image
        try:
            dart_score = score(full_path, cfgFile="holo_v1")
            logger.info(f"Image processed: {full_path}, Score: {dart_score}")
            return str(dart_score)  # Return as string for easy parsing
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return '-1', 500
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True) 