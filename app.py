from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import os
import csv
from flask_cors import CORS

app = Flask(__name__, static_folder='static')  # Set the static folder for your HTML, CSS, and JS files
CORS(app)

# Paths to real and fake image directories
REAL_DIR = './real_fake_images/real'
FAKE_DIR = './real_fake_images/fake'

@app.route('/')
def serve_index():
    """
    Serve the index.html file.
    """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """
    Serve static files (HTML, CSS, JS).
    """
    return send_from_directory(app.static_folder, filename)

# Add the other routes you already have
@app.route('/get-images', methods=['GET'])
def get_images():
    try:
        real_images = os.listdir(REAL_DIR)
        fake_images = os.listdir(FAKE_DIR)
        return jsonify({
            'real_images': real_images,
            'fake_images': fake_images
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    ai_selection = data.get('aiSelection')  # 'img1' or 'img2'
    quality = data.get('quality')  # Quality rating
    image1 = data.get('image1')  # Image 1 file name
    image2 = data.get('image2')  # Image 2 file name
    timestamp = datetime.now().isoformat()

    # Determine the non-selected image
    non_selected_image = image1 if ai_selection == 'img2' else image2

    # Save to CSV
    with open('responses.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, image1, image2, ai_selection, non_selected_image, quality])

    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
