from flask import Flask, request, jsonify,render_template
from werkzeug.utils import secure_filename
import os
from src.brain_tumor_classifier.components.model_predict import Model_Predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    predictor = Model_Predict()
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction = predictor.initiate_prediction_pipeline(file_path)
        if prediction == 0:
            prediction = 'Glioma Tumor'
        elif prediction == 1:
            prediction = 'Meningioma Tumor'
        elif prediction == 2:
            prediction = 'No Tumor'
        elif prediction == 3:
            prediction = 'Pituitary Tumor'
        
        return jsonify({'prediction': prediction})
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
