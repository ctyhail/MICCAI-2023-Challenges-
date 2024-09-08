from flask import Flask, request, jsonify,send_from_directory
from PIL import Image
from flask_cors import CORS  # 导入 CORS
import io
import os
import numpy as np
import cv2
from infer import detect

app = Flask(__name__,static_folder='results')
CORS(app)  # 启用 CORS 支持

# 保存上传图像的目录
UPLOAD_FOLDER = r'uploads'
RESULT_FOLDER = r'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)  # 保存上传的图像
        
        result_image = detect(file_path)
        result_image_path = os.path.join(r'D:\暑期考核\results', file.filename)
        result_image.save(result_image_path)

        
        return jsonify({'resultImageUrl': result_image_path})

    return jsonify({'error': 'Invalid file format'}), 400

def allowed_file(filename):
    return filename.lower().endswith(('png', 'jpg', 'jpeg'))

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)