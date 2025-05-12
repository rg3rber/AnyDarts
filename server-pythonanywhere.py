from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
import numpy as np

""" Simplified Flask server for hosted image upload and processing """

app = Flask(__name__)
from DartsPrediction import score

CORS(app)

@app.route('/', methods=['POST'])
def handle_image_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img_data = file.read()
        img_bytesio = BytesIO(img_data)
        
        # Attempt to score the image
        try:
            img = Image.open(img_bytesio)
            img_array = np.array(img)
            dart_score = score(img_array, cfgFile="holo_v1")
            #logger.info(f"Image processed: {full_path}, Score: {dart_score}")
            return str(dart_score)  # Return as string for easy parsing
        
        except Exception as e:
            #logger.error(f"Scoring error: {e}")
            return '-1', 500
    
    except Exception as e:
        #logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 