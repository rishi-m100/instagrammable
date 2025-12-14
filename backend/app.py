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

# Load CLIP (Quantized + Split)
# Reassemble split file if needed
quantized_weights_path = "clip_full_quantized.pth"
if not os.path.exists(quantized_weights_path):
    print("🧩 Reassembling split model files...")
    with open(quantized_weights_path, 'wb') as output_file:
        chunk_num = 0
        while True:
            chunk_name = f"{quantized_weights_path}.part{chunk_num}"
            if not os.path.exists(chunk_name):
                break
            with open(chunk_name, 'rb') as chunk_file:
                output_file.write(chunk_file.read())
            chunk_num += 1
    print("✅ Model reassembled.")

# Initialize empty model structure
print("📉 Loading Quantized CLIP Model Object...")
torch.backends.quantized.engine = 'qnnpack'
# Load the ENTIRE pickled object (structure + weights)
# This avoids initializing the 600MB float32 model first
# Note: weights_only=False is required because we are loading a full model object (code), not just weights.
clip_model = torch.load(quantized_weights_path, map_location=device, weights_only=False)

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
    port = int(os.environ.get("PORT", 5002))
    app.run(port=port, debug=True)