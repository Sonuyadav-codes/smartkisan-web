import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import gc
import traceback
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from predict import predict_disease_api

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TREATMENTS = {
    "bacterial_leaf_blight": "Apply copper-based bactericide.\nAvoid excess nitrogen fertilizer.",
    "brown_spot": "Spray recommended fungicide (e.g., Mancozeb).\nMaintain proper soil nutrients & field drainage.",
    "healthy": "The rice leaf is healthy.\nNo chemical treatment required. Keep monitoring.",
    "leaf_blast": "Apply systemic fungicide (e.g., Tricyclazole).\nAvoid excessive use of nitrogenous fertilizers.",
    "leaf_scald": "Apply recommended fungicide.\nMaintain proper field sanitation & clean seeds.",
    "narrow_brown_spot": "Apply potassium fertilizers and suitable fungicides.\nEnsure good field drainage."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    filepath = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid image file type. Upload JPG, JPEG or PNG.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run Prediction Pipeline
        result = predict_disease_api(filepath)

        disease = str(result.get('disease', 'Invalid Image'))
        confidence_str = str(result.get('confidence', '0.0 %'))
        confidence_val = float(result.get('confidence_val', 0.0))

        disease_key = disease.lower().strip()
        
        if disease_key == "invalid image" or confidence_val < 40.0:
            treatment = "Unable to identify as a valid rice leaf.\nPlease upload or capture a clear rice leaf image."
        else:
            treatment = TREATMENTS.get(disease_key, "Treatment details not available for this disease.")

        return jsonify({
            'image_url': f'/static/uploads/{filename}',
            'disease': disease,
            'confidence': confidence_str,
            'confidence_val': confidence_val,
            'treatment': treatment
        })

    except Exception as e:
        print("---- BACKEND ERROR ----", file=sys.stderr)
        traceback.print_exc()
        return jsonify({'error': f'Server Error: {str(e)}'}), 500

    finally:
        # Prevent RAM & Disk Overload on Render Cloud
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        gc.collect()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
