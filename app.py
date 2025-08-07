import yaml
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Model & Training Imports (from your project) ---
from pretrain import PretrainConfig, create_model
from puzzle_dataset import PuzzleDatasetMetadata

# ---------------------------
# 0. Device Configuration
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Using device: {DEVICE}")

# ---------------------------
# 1. Configuration
# ---------------------------
# --- IMPORTANT ---
# Update this path to point to your trained model checkpoint
CHECKPOINT_PATH = "checkpoints/City-logistics-1k ACT-torch/HierarchicalReasoningModel_ACTV1 peach-corgi/step_23430"
# --- IMPORTANT ---

# Load the configuration file that was saved during training
config_path = f"{'/'.join(CHECKPOINT_PATH.split('/')[:-1])}/all_config.yaml"
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)
    # We don't need a real data_path for inference, so we can create a dummy one
    config_dict['data_path'] = 'data/logistics-routing-1k' 
    config = PretrainConfig(**config_dict)

# Create a dummy metadata object for model initialization
# These values should match what was used in training
dummy_metadata = PuzzleDatasetMetadata(
    pad_id=0,
    ignore_label_id=-100,
    blank_identifier_id=0,
    vocab_size=10,
    seq_len=1600,
    num_puzzle_identifiers=1,
    total_groups=1,
    mean_puzzle_examples=1.0,
    sets=['all']
)

# ---------------------------
# 2. Model Loading
# ---------------------------
print(">>> Loading model...")
# Initialize the model structure
model, _, _ = create_model(config, dummy_metadata, world_size=1)

# Load the trained weights from your checkpoint
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE) # Ensure model is on the correct device
model.eval()  # Set the model to evaluation mode
print(">>> Model loaded successfully!")

# ---------------------------
# 3. Path Decoding Functions
# ---------------------------
MAP_DIMENSIONS = {"width": 30, "height": 30}

HRM_TOKEN_MAP = {
    "PAD": 0,           # Padding token
    "OBSTACLE": 1,      # Buildings/City Park
    "SMALL_ROAD": 2,    # Side Streets  
    "LARGE_ROAD": 3,    # Major Avenues
    "DIAGONAL": 4,      # Main diagonal thoroughfare
    "TRAFFIC_JAM": 5,   # Heavy Traffic
    "ROAD_CLOSURE": 6,  # Road Closure
    "START": 7,         # Start Point
    "END": 8,           # End Point
    "PATH": 9           # Optimal Route
}

def decode_path_from_logits(logits):
    """Extract path coordinates from model logits"""
    # Get predicted tokens
    predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
    
    # Extract PATH tokens (token value 9)
    path_coords = []
    path_token_count = 0
    
    for i, token in enumerate(predicted_tokens):
        if token == HRM_TOKEN_MAP["PATH"]:
            path_token_count += 1
            y = i // MAP_DIMENSIONS["width"]
            x = i % MAP_DIMENSIONS["width"]
            path_coords.append({"x": int(x), "y": int(y)})
    
    print(f"   🔍 Debug: Found {path_token_count} PATH tokens, {len(path_coords)} coordinates")
    
    # If we have too many PATH tokens, the model might be outputting incorrectly
    if len(path_coords) > 200:  # Reasonable maximum for a 40x40 grid
        print(f"   ⚠️  WARNING: Model outputting too many PATH tokens ({len(path_coords)})")
        print(f"   ⚠️  This suggests the model hasn't learned proper pathfinding")
        
        # Try to extract a reasonable path by filtering
        # Look for connected components or use other heuristics
        return extract_reasonable_path(path_coords)
    
    return path_coords

def extract_reasonable_path(all_coords):
    """Extract a reasonable path when model outputs too many PATH tokens"""
    if len(all_coords) == 0:
        return []
    
    # Simple heuristic: find the longest connected path
    # This is a fallback when the model hasn't learned properly
    print(f"   🔧 Applying fallback path extraction...")
    
    # For now, return a subset - in a real scenario you'd implement
    # more sophisticated path extraction
    return all_coords[:50]  # Limit to reasonable size

def extract_start_end_from_logits(logits):
    """Extract start and end positions from logits"""
    predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
    
    start_pos = None
    end_pos = None
    
    for i, token in enumerate(predicted_tokens):
        y = i // MAP_DIMENSIONS["width"]
        x = i % MAP_DIMENSIONS["width"]
        
        if token == HRM_TOKEN_MAP["START"]:
            start_pos = {"x": int(x), "y": int(y)}
        elif token == HRM_TOKEN_MAP["END"]:
            end_pos = {"x": int(x), "y": int(y)}
    
    return start_pos, end_pos

# ---------------------------
# 4. Flask Server Setup
# ---------------------------
app = Flask(__name__)
# CORS is required to allow your local HTML file to make requests to this local server
CORS(app) 

@app.route("/predict", methods=["POST"])
def predict():
    """
    Enhanced prediction endpoint that logs intermediate reasoning paths
    """
    # Get the input sequence from the request
    data = request.get_json()
    input_sequence = data.get("inputs")
    log_intermediate = data.get("log_intermediate", True)  # New parameter

    if not input_sequence:
        return jsonify({"error": "No input sequence provided"}), 400

    # Convert the input to a PyTorch tensor
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # --- Run Model Inference with Logging ---
    intermediate_paths = []
    reasoning_steps = []
    
    with torch.no_grad():
        # Create a dummy batch structure
        dummy_batch = {
            "inputs": input_tensor,
            "labels": torch.zeros_like(input_tensor),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long).to(DEVICE)
        }
        
        # Initialize carry state
        carry = model.initial_carry(dummy_batch)
        
        # --- CORRECTED & ROBUST CARRY-TO-DEVICE LOGIC ---
        def move_to_device(obj, device):
            if hasattr(obj, 'to'):
                return obj.to(device)
            if isinstance(obj, (list, tuple)):
                return type(obj)(move_to_device(x, device) for x in obj)
            if isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            if hasattr(obj, '__dict__'):
                for attr, value in obj.__dict__.items():
                    setattr(obj, attr, move_to_device(value, device))
            return obj

        carry = move_to_device(carry, DEVICE)
        
        # Enhanced reasoning loop with logging
        step_count = 0
        while True:
            # Forward pass
            carry, _, _, outputs, all_halted = model(return_keys=["logits"], carry=carry, batch=dummy_batch)
            
            if log_intermediate and "logits" in outputs:
                # Extract intermediate path - take only first example
                intermediate_path = decode_path_from_logits(outputs["logits"][0:1])
                start_pos, end_pos = extract_start_end_from_logits(outputs["logits"][0:1])
                
                # Log this reasoning step
                step_info = {
                    "step": int(step_count),
                    "path": intermediate_path,
                    "path_length": int(len(intermediate_path)),
                    "start": start_pos,
                    "end": end_pos,
                    "halted": bool(all_halted)
                }
                reasoning_steps.append(step_info)
                
                print(f"🧠 HRM Step {step_count}: Found path with {len(intermediate_path)} waypoints")
                if len(intermediate_path) > 0:
                    print(f"   Path preview: {intermediate_path[:3]}{'...' if len(intermediate_path) > 3 else ''}")
            
            step_count += 1
            
            if all_halted:
                break
        
        # Get final prediction - take only the first example (they're all identical)
        final_logits = outputs["logits"]
        prediction = torch.argmax(final_logits, dim=-1)
        predicted_path_sequence = prediction[0].cpu().tolist()  # Take first example only
        final_path_coords = decode_path_from_logits(final_logits[0:1])  # Process first example only

    print(f"🎯 HRM completed reasoning in {step_count} steps")
    print(f"📍 Final path: {len(final_path_coords)} waypoints")

    # Return enhanced result - ensure all values are JSON serializable
    result = {
        "path": [int(x) for x in predicted_path_sequence],  # Convert to Python ints
        "path_coords": final_path_coords,  # Already converted above
        "reasoning_steps": reasoning_steps if log_intermediate else [],
        "total_steps": int(step_count),
        "final_path_length": int(len(final_path_coords))
    }
    
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True})

# ---------------------------
# 5. Run the Server
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting HRM inference server with path logging...")
    print("📊 Intermediate reasoning steps will be logged to console")
    print("🌐 API available at: http://127.0.0.1:5000")
    print("📝 Endpoints:")
    print("   POST /predict - Get model predictions with reasoning logs")
    print("   GET  /health  - Health check")
    
    # This makes the server accessible on your local machine at http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=False)