import os
import pickle
import numpy as np
import shutil

# --- Monkeypatching Logic ---
# The issue is that the models point to 'numpy.random._mt19937.MT19937'
# but in this numpy version it is accessed/validated differently.
# We intercept the unpickler's constructor for bit generators.

import numpy.random._pickle

original_bit_generator_ctor = numpy.random._pickle.__bit_generator_ctor

def patched_bit_generator_ctor(bit_generator_name):
    # If the name is the problematic class, return the string name 'MT19937'
    # which the original function knows how to handle (usually).
    if str(bit_generator_name) == "<class 'numpy.random._mt19937.MT19937'>":
        return original_bit_generator_ctor("MT19937")
    return original_bit_generator_ctor(bit_generator_name)

# Apply patch
numpy.random._pickle.__bit_generator_ctor = patched_bit_generator_ctor

MODELS_DIR = "./models"
BACKUP_DIR = "./models_backup"

def fix_models():
    if not os.path.exists(MODELS_DIR):
        print("❌ No models directory found.")
        return

    # Create backup
    if not os.path.exists(BACKUP_DIR):
        print(f"📦 Creating backup in {BACKUP_DIR}...")
        shutil.copytree(MODELS_DIR, BACKUP_DIR)
    else:
        print(f"ℹ️ Backup directory {BACKUP_DIR} already exists. Skipping backup.")

    print("🔧 Starting model repair...")
    
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    
    for filename in files:
        filepath = os.path.join(MODELS_DIR, filename)
        print(f"  Processing {filename}...", end=" ")
        
        try:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            
            # Re-save with current numpy version
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
            
            print("✅ Fixed & Re-saved")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

    print("\n🎉 Repair process complete. Try running app.py now.")

if __name__ == "__main__":
    fix_models()
