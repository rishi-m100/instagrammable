import os
import pickle
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)
CORS(app)

# --- GLOBAL VARIABLES (initially None) ---
clip_model = None
processor = None
trained_models = {}
MODELS_LOADED = False # Flag to track status

# --- HELPER: Lazy Loader Function ---
def load_models_lazy():
    global clip_model, processor, trained_models, MODELS_LOADED
    
    if MODELS_LOADED:
        return # Already loaded, skip

    print("⏳ Starting Lazy Load of Models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load CLIP
    quantized_weights_path = "clip_full_quantized.pth"
    # (Reassembly logic skipped for brevity, assuming file exists from previous runs)
    
    print("📉 Loading CLIP...")
    torch.backends.quantized.engine = 'qnnpack'
    # Load CLIP (heavy!)
    clip_model = torch.load(quantized_weights_path, map_location=device, weights_only=False)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 2. Load Sklearn Models
    MODELS_DIR = "./models"
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
    
    MODELS_LOADED = True
    print("🚀 All Models Loaded Successfully!")

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    # This route is lightweight and won't crash the server
    return jsonify({
        "status": "running", 
        "message": "Server is active. Models load on first request.",
        "models_ready": MODELS_LOADED
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    # TRIGGER LOAD HERE (The first user waits a few seconds, but server won't crash on boot)
    if not MODELS_LOADED:
        load_models_lazy()

    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for file in files:
        try:
            image = Image.open(file.stream)
            image = ImageOps.exif_transpose(image).convert("RGB")
            
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            embedding = features.cpu().numpy().reshape(1, -1)
            
            model_scores = {}
            total_score = 0
            
            for name, model in trained_models.items():
                pred = model.predict(embedding)[0]
                if "Classifier" not in name:
                    pred = max(0.0, min(5.0, pred))
                model_scores[name] = round(float(pred), 2)
                total_score += pred
            
            avg_score = total_score / len(trained_models) if trained_models else 0
            
            results.append({
                "filename": file.filename,
                "average_score": round(avg_score, 2),
                "breakdown": model_scores
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(port=port, debug=True)