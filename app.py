from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='static')  # Set static folder for HTML, CSS, JS
CORS(app)

# PostgreSQL database connection from Render
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://annotations_mhjf_user:ePceyPCY8SNNKVfVwtkP03fgf0QDXL66@dpg-cuhtnoggph6c73boa1og-a.ohio-postgres.render.com/annotations_mhjf')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define the database model (replaces CSV storage)
class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image1 = db.Column(db.String(255), nullable=False)
    image2 = db.Column(db.String(255), nullable=False)
    ai_selection = db.Column(db.String(10), nullable=False)
    non_selected_image = db.Column(db.String(255), nullable=False)
    quality = db.Column(db.Integer, nullable=False)

# Create the database tables if they don't exist
with app.app_context():
    db.create_all()

# Serve the homepage
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Serve static files (HTML, CSS, JS)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Serve images from real_fake_images folder
@app.route('/real_fake_images/<folder>/<filename>')
def serve_images(folder, filename):
    return send_from_directory(f'./real_fake_images/{folder}', filename)

# Return a list of images from the real and fake directories
@app.route('/get-images', methods=['GET'])
def get_images():
    try:
        real_images = os.listdir('./real_fake_images/real')
        fake_images = os.listdir('./real_fake_images/fake')
        return jsonify({'real_images': real_images, 'fake_images': fake_images}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Store the form submission in the database instead of CSV
@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    ai_selection = data.get('aiSelection')  # 'img1' or 'img2'
    quality = data.get('quality')  # Quality rating
    image1 = data.get('image1')  # Image 1 file name
    image2 = data.get('image2')  # Image 2 file name

    # Determine the non-selected image
    non_selected_image = image1 if ai_selection == 'img2' else image2

    # Save response to PostgreSQL
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

# Fetch stored responses from PostgreSQL (view responses)
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

# Start the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
