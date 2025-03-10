import os
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

app = Flask(__name__, static_folder='static')
CORS(app)

DATABASE_URL = os.getenv('DATABASE_URL')
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

# ----------------- PyTorch Model Setup -------------------

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A simple CNN architecture (expects 224x224 RGB images)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # After two poolings, 224 becomes 56 (224/2/2 = 56)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes: 0=fake, 1=real
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize model and load pretrained weights if available
model = SimpleCNN().to(device)
if os.path.exists('cnn_model.pth'):
    model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
model.eval()

# Define optimizer and loss for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize using typical ImageNet stats (adjust if needed)
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_and_preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

def pytorch_predict(image1_path, image2_path):
    model.eval()
    with torch.no_grad():
        img1 = load_and_preprocess(image1_path)
        img2 = load_and_preprocess(image2_path)
        # Forward pass for both images
        output1 = model(img1)
        output2 = model(img2)
        # Softmax probabilities (we consider index 1 as the probability for "real")
        prob1 = torch.softmax(output1, dim=1)[0, 1].item()
        prob2 = torch.softmax(output2, dim=1)[0, 1].item()
        # The image with the higher probability is predicted as real.
        if prob1 > prob2:
            prediction = "img1"
            confidence = prob1
        else:
            prediction = "img2"
            confidence = prob2
    return prediction, confidence

def fine_tune_with_sample(real_image_path, fake_image_path):
    # Switch model to train mode
    model.train()
    real_img = load_and_preprocess(real_image_path)
    fake_img = load_and_preprocess(fake_image_path)
    # Get model outputs
    output_real = model(real_img)
    output_fake = model(fake_img)
    # Set targets: real image should have label 1, fake image label 0.
    target_real = torch.tensor([1]).to(device)
    target_fake = torch.tensor([0]).to(device)
    loss_real = criterion(output_real, target_real)
    loss_fake = criterion(output_fake, target_fake)
    loss = (loss_real + loss_fake) / 2.0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    # Save updated model state for persistence
    torch.save(model.state_dict(), 'cnn_model.pth')
    return loss.item()

# ------------------ Flask Endpoints ----------------------

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/real_fake_images/<folder>/<filename>')
def serve_images(folder, filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(os.path.join(base_dir, 'real_fake_images', folder), filename)

@app.route('/get-images', methods=['GET'])
def get_images():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        real_path = os.path.join(base_dir, 'real_fake_images', 'real')
        fake_path = os.path.join(base_dir, 'real_fake_images', 'fake')
        real_images = os.listdir(real_path)
        fake_images = os.listdir(fake_path)
        return jsonify({'real_images': real_images, 'fake_images': fake_images}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Prediction endpoint that compares two images
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image1 = data.get('image1')
    folder1 = data.get('folder1')
    image2 = data.get('image2')
    folder2 = data.get('folder2')
    if not image1 or not folder1 or not image2 or not folder2:
        return jsonify({'error': 'Missing image filenames or folder info'}), 400
    image1_path = os.path.join('real_fake_images', folder1, image1)
    image2_path = os.path.join('real_fake_images', folder2, image2)
    try:
        prediction, confidence = pytorch_predict(image1_path, image2_path)
        return jsonify({'prediction': prediction, 'confidence': confidence}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Submission endpoint that logs user selections and fine-tunes the model
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
        folder1 = ann.get('folder1')  # Make sure your front-end sends this info
        image2 = ann.get('image2')
        folder2 = ann.get('folder2')
        
        # Determine the non-selected image (for record keeping)
        non_selected_image = image1 if ai_selection == 'img2' else image2
        
        response_entry = Response(
            image1=image1,
            image2=image2,
            ai_selection=ai_selection,
            non_selected_image=non_selected_image,
            quality=int(quality)
        )
        db.session.add(response_entry)
        
        # Fine-tune with every sample:
        # If user selects img1 as real, then image1 is real and image2 is fake; otherwise vice versa.
        if ai_selection == 'img1':
            real_path = os.path.join('real_fake_images', folder1, image1)
            fake_path = os.path.join('real_fake_images', folder2, image2)
        else:
            real_path = os.path.join('real_fake_images', folder2, image2)
            fake_path = os.path.join('real_fake_images', folder1, image1)
        try:
            loss = fine_tune_with_sample(real_path, fake_path)
            print(f"Fine-tuning loss: {loss}")
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
    
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
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
