import torch
import os
from transformers import CLIPModel

def compress_and_split():
    # Force QNNPACK (required for quantization on some storage backends)
    torch.backends.quantized.engine = 'qnnpack'
    
    print("⏳ Loading CLIP model (this might take a moment)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    print("📉 Quantizing model (Float32 -> Int8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save optimized state dict
    output_path = "clip_quantized.pth"
    print(f"💾 Saving to {output_path}...")
    torch.save(quantized_model.state_dict(), output_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ quantizing complete! New size: {size_mb:.2f} MB")
    
    # Split into chunks (Github limit is 100MB)
    chunk_size = 90 * 1024 * 1024 # 90MB
    chunk_num = 0
    
    with open(output_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            chunk_name = f"{output_path}.part{chunk_num}"
            with open(chunk_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"📦 Created chunk: {chunk_name}")
            chunk_num += 1
            
    # Cleanup original huge file
    os.remove(output_path)
    print("🧹 Cleaned up temporary large file.")
    print("🎉 Done! Push these .part files to GitHub.")

if __name__ == "__main__":
    compress_and_split()
