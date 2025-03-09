from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

DATABASE_URL = os.getenv('DATABASE_URL')

# Ensure SSL mode is required for Neon.tech
if DATABASE_URL and "sslmode=" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image1 = db.Column(db.String(255), nullable=False)
    image2 = db.Column(db.String(255), nullable=False)
    ai_selection = db.Column(db.String(10), nullable=False)
    non_selected_image = db.Column(db.String(255), nullable=False)
    quality = db.Column(db.Integer, nullable=False)

with app.app_context():
    db.create_all()

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/real_fake_images/<folder>/<filename>')
def serve_images(folder, filename):
    return send_from_directory(f'./real_fake_images/{folder}', filename)

@app.route('/get-images', methods=['GET'])
def get_images():
    try:
        real_images = os.listdir('./real_fake_images/real')
        fake_images = os.listdir('./real_fake_images/fake')
        return jsonify({'real_images': real_images, 'fake_images': fake_images}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    annotations = data.get('annotations')
    if not annotations:
        return jsonify({'error': 'No annotations provided'}), 400
    
    for ann in annotations:
        ai_selection = ann.get('aiSelection')
        quality = ann.get('quality')
        image1 = ann.get('image1')
        image2 = ann.get('image2')

        # Determine the non-selected image
        non_selected_image = image1 if ai_selection == 'img2' else image2

        response = Response(
            image1=image1,
            image2=image2,
            ai_selection=ai_selection,
            non_selected_image=non_selected_image,
            quality=int(quality)
        )
        db.session.add(response)

    db.session.commit()
    return jsonify({'status': 'success'}), 200


@app.route('/responses', methods=['GET'])
def get_responses():
    responses = Response.query.all()
    result = [{
        'id': response.id,
        'timestamp': response.timestamp.isoformat(),
        'image1': response.image1,
        'image2': response.image2,
        'ai_selection': response.ai_selection,
        'non_selected_image': response.non_selected_image,
        'quality': response.quality
    } for response in responses]
    
    return jsonify(result), 200

@app.route('/clear', methods=['POST'])
def clear():
    try:
        db.session.query(Response).delete()
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'rows deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
