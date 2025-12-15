import os
import pickle
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)  # Allow React to communicate with Flask

# --- 1. Load Models (Run once on startup) ---
print("⏳ Loading AI Models... this may take a minute.")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- EXPLAINABILITY SETUP ---
genai.configure(api_key="AIzaSyA5n3Ha038s1LJ2FYmlmhE3h59WAgZgnUs")

concept_list = [
    # --- POSITIVE: COMPOSITION & TECH ---
    "minimalist", "flat lay", "rule of thirds", "symmetry", "leading lines",
    "depth of field", "bokeh", "sharp focus", "clean lines", "balanced composition",
    "negative space", "framed subject", "cinematic", "wide angle", "macro detail",
    "perspective", "geometric patterns", "high resolution", "professional grading",

    # --- POSITIVE: LIGHTING & COLOR ---
    "golden hour", "natural light", "soft lighting", "high contrast", "vibrant colors",
    "pastel aesthetic", "neon lights", "warm tones", "cool tones", "monochrome",
    "black and white", "saturated", "muted tones", "shadow play", "sun flare",
    "dramatic lighting", "silhouette", "airy", "bright and airy", "moody lighting",

    # --- POSITIVE: VIBE & AESTHETIC ---
    "luxury", "cozy", "rustic", "vintage", "retro", "futuristic",
    "urban", "industrial", "nature", "tropical", "beach vibes",
    "romantic", "dreamy", "ethereal", "energetic", "peaceful",
    "sophisticated", "elegant", "edgy", "whimsical", "nostalgic",
    "clean girl aesthetic", "cottagecore", "dark academia", "y2k aesthetic",
    "streetwear", "old money aesthetic", "travel wanderlust", "mindfulness",

    # --- SUBJECT SPECIFIC ---
    "portrait", "candid moment", "group photo", "selfie", "outfit of the day",
    "food styling", "coffee art", "interior design", "architecture", "landscape",
    "cityscape", "night life", "festival", "gym fitness", "working from home",
    "plant parent", "pet photography", "product showcase", "texture",

    # --- NEGATIVE: TECHNICAL FLAWS ---
    "blurry", "grainy", "pixelated", "low resolution", "out of focus",
    "motion blur", "camera shake", "noise", "artifacts", "chromatic aberration",
    "lens flare", "distorted", "overprocessed", "unnatural filters",

    # --- NEGATIVE: LIGHTING & COMPOSITION ---
    "dark lighting", "underexposed", "overexposed", "washed out", "flash glare",
    "harsh shadows", "poor lighting", "yellow cast", "blue cast",
    "cluttered", "messy", "chaotic background", "distracting elements",
    "crooked horizon", "bad framing", "awkward angle", "crowded",
    "boring composition", "flat lighting", "dull colors"
]

print("Encoding concept bank for explainability...")
text_features = None

# Encode them immediately so they are ready for comparison
try:
    text_inputs = processor(text=concept_list, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, p=2, dim=1)
    print(f"✅ Concept bank encoded: {len(concept_list)} terms.")
except Exception as e:
    print(f"❌ Failed to encode concepts: {e}")


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


def get_explanation_data(image_features, score):
    """
    1. Finds top matching concepts from CLIP.
    2. Sends data to Gemini to get a human-readable explanation.
    """
    if text_features is None:
        return [], "Explainability module not loaded."

    # --- 1. CLIP Concept Matching ---
    # Calculate Cosine Similarity
    sim = (image_features @ text_features.T).squeeze(0)
    
    # Get top 5 matching words for better context
    values, indices = sim.topk(5) 
    top_words = [concept_list[idx] for idx in indices]
    
    # --- 2. Construct Prompt for Gemini ---
    prompt = f"""
    You are an expert social media photographer and aesthetic critic.
    An AI model has analyzed an image and given it an 'Instagrammable Score'.
    
    Data:
    - **Score:** {score:.2f}/5.0
    - **Detected Visual Features:** {', '.join(top_words)}
    
    Task:
    Write a 2-sentence explanation for the user.
    - If the score is high (>4.0), explain why it's great using the detected features.
    - If the score is low (<2.5), explain what is hurting the score (e.g. "The low score is likely due to...").
    - If the score is average, mention the good and bad.
    
    Keep it direct and helpful. Do not mention "CLIP" or "vectors".
    """
    
    # --- 3. Call Gemini API ---
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        
        # FIX: Extract text from the response properly
        # response.text may throw an error if there are content safety issues
        try:
            explanation = response.text.strip()
        except:
            # If response.text fails, extract from content parts
            if response.parts:
                explanation = response.parts[0].text.strip()
            else:
                explanation = "Unable to generate explanation at this time."
            
    except Exception as e:
        explanation = f"Unable to generate explanation at this time."
        print(f"Gemini API Error: {e}")
        
    return top_words, explanation

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
                if "Classifier" in name:
                    # 1. Use predict_proba for smooth scoring on the classifier
                    probs = model.predict_proba(embedding)[0]
                    classes = model.classes_
                    # Calculate weighted average: (prob_of_1 * 1) + (prob_of_2 * 2) ...
                    pred = sum(p * c for p, c in zip(probs, classes))
                else:
                    pred = model.predict(embedding)[0]
                pred = max(0.0, min(5.0, pred)) # Clip 0-5
                
                model_scores[name] = round(float(pred), 2)
                total_score += pred
            
            avg_score = total_score / len(trained_models) if trained_models else 0
            
            # --- Get Explainability Data ---
            top_concepts, explanation = get_explanation_data(features, avg_score)

            results.append({
                "filename": file.filename,
                "average_score": round(avg_score, 2),
                "breakdown": model_scores,
                "concepts": top_concepts,
                "explanation": explanation
            })
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5001, debug=True)