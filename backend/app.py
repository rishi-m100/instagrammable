import os
import pickle
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)
CORS(app)  # Allow React to communicate with Flask

# --- 1. Load Models (Run once on startup) ---
print("⏳ Loading AI Models... this may take a minute.")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load Your Trained Ensemble Models
MODELS_DIR = "./models"
trained_models = {}

if os.path.exists(MODELS_DIR):
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pkl"):
            name = filename.replace("model_", "").replace(".pkl", "").replace("_", " ")
            try:
                with open(os.path.join(MODELS_DIR, filename), "rb") as f:
                    trained_models[name] = pickle.load(f)
                print(f"✅ Loaded: {name}")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
else:
    print("⚠️ WARNING: No 'models' folder found. Please create it and add your .pkl files.")

print("🚀 Server is ready!")

# --- 2. Prediction Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    results = []
    
    # Process images
    # Note: For production, we would process in batches. For a simple app, loop is fine.
    for file in files:
        try:
            # Load and Preprocess
            image = Image.open(file.stream)
            image = ImageOps.exif_transpose(image).convert("RGB")
            
            # CLIP Inference
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                # Normalize (Critical for cosine similarity / aligned training)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            embedding = features.cpu().numpy().reshape(1, -1)
            
            # Ensemble Prediction
            model_scores = {}
            total_score = 0
            
            for name, model in trained_models.items():
                pred = model.predict(embedding)[0]
                
                # Handle Classifier vs Regressor logic
                if "Classifier" not in name:
                    pred = max(0.0, min(5.0, pred)) # Clip 0-5
                
                model_scores[name] = round(float(pred), 2)
                total_score += pred
            
            avg_score = total_score / len(trained_models) if trained_models else 0
            
            results.append({
                "filename": file.filename,
                "average_score": round(avg_score, 2),
                "breakdown": model_scores
            })
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(port=port, debug=True)