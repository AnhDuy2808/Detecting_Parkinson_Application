# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os
from src.extract_features import extract_features
from pydub import AudioSegment

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải mô hình đã được huấn luyện
MODEL_DIR = './model_assets_voting'
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'voting_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'r') as f:
        feature_columns = json.load(f)
    print("Tải mô hình Ensemble thành công.")
except FileNotFoundError:
    print("LỖI: Không tìm thấy tệp mô hình. Vui lòng chạy 'notebook/parkinson_classification.ipynb' trước.")
    model = None


@app.route('/')
def index():
    return render_template('index.html', features=feature_columns)


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'Không có tệp âm thanh'}), 400

    audio_file = request.files['audio_data']
    
    original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_audio_original")
    wav_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_audio_converted.wav")

    try:
        audio_file.save(original_file_path)

        # Chuẩn hóa file âm thanh về định dạng WAV
        sound = AudioSegment.from_file(original_file_path)
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(16000)
        sound.export(wav_file_path, format="wav")

        # Trích xuất đặc trưng từ file âm thanh
        cls_model_dir = './model_assets_stacking'
        cls_model = joblib.load(os.path.join(cls_model_dir, 'stacking_model.joblib'))
        cls_scaler = joblib.load(os.path.join(cls_model_dir, 'scaler.joblib'))
        with open(os.path.join(cls_model_dir, 'feature_columns.json'), 'r') as f:
            cls_feature_names = json.load(f)
        
        cls_features = extract_features(wav_file_path) 
        
        if cls_features is None or not cls_features:
            return jsonify({'error': 'Không thể trích xuất đặc trưng từ âm thanh. Vui lòng thử ghi âm lại rõ ràng và dài hơn.'}), 500

        # Chuẩn bị dữ liệu và đưa ra dự đoán
        df_cls = pd.DataFrame([cls_features])
        df_cls = df_cls[cls_feature_names]
        X_cls_scaled = cls_scaler.transform(df_cls)
        
        # Lấy xác suất của cả 2 lớp
        probabilities = cls_model.predict_proba(X_cls_scaled)[0]
        
        healthy_prob = probabilities[0] * 100
        parkinson_prob = probabilities[1] * 100

        # Tạo kết quả trả về dưới dạng JSON
        result = {
            'healthy_percentage': f"{healthy_prob:.2f}%",
            'parkinson_percentage': f"{parkinson_prob:.2f}%"
        }
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Đã xảy ra lỗi trong quá trình xử lý.'}), 500

    finally:
        # Xóa file tạm
        if os.path.exists(original_file_path): 
            os.remove(original_file_path)
        if os.path.exists(wav_file_path): 
            os.remove(wav_file_path)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)