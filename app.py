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
CHECKPOINT_PATH = "checkpoints/Logistics-routing-1k ACT-torch/HierarchicalReasoningModel_ACTV1 rigorous-mushroom/step_310"
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
# 3. Flask Server Setup
# ---------------------------
app = Flask(__name__)
# CORS is required to allow your local HTML file to make requests to this local server
CORS(app) 

@app.route("/predict", methods=["POST"])
def predict():
    """
    This function is the API endpoint. It receives the map data,
    runs the model, and returns the predicted path.
    """
    # Get the input sequence from the request
    data = request.get_json()
    input_sequence = data.get("inputs")

    if not input_sequence:
        return jsonify({"error": "No input sequence provided"}), 400

    # Convert the input to a PyTorch tensor
    # The model expects a batch, so we add a batch dimension (unsqueeze)
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # --- Run Model Inference ---
    with torch.no_grad():
        # Create a dummy batch structure similar to what the model expects
        # The model was trained with puzzle_identifiers, so we need to provide a dummy one
        dummy_batch = {
            "inputs": input_tensor,
            # The model expects labels for its internal logic, even during inference. We can provide a dummy tensor.
            "labels": torch.zeros_like(input_tensor),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long).to(DEVICE) # Dummy identifier
        }
        
        # The model uses a 'carry' state for its recurrent nature. We initialize it.
        carry = model.initial_carry(dummy_batch)
        
        # --- FIX: Manually move all tensors in the carry state to the correct device ---
        # This is necessary because the model's initial_carry function creates some tensors on the CPU by default.
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(DEVICE)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(DEVICE)
        carry.steps = carry.steps.to(DEVICE)
        carry.halted = carry.halted.to(DEVICE)
        # carry.current_data is already on the correct device due to torch.empty_like()
        
        # The model might take multiple steps to "think". We loop until it has halted.
        while True:
            # The model is wrapped in ACTLossHead, which has a different signature.
            # We must call it with keyword arguments and handle its 5 return values.
            carry, _, _, outputs, all_halted = model(return_keys=["logits"], carry=carry, batch=dummy_batch)
            if all_halted:
                break
        
        # Get the final prediction (logits) from the outputs dictionary
        logits = outputs["logits"]
        # The predicted path is the index with the highest value in the logits
        prediction = torch.argmax(logits, dim=-1)
        
        # The output is a tensor on the GPU, move it to the CPU and convert to a list
        predicted_path_sequence = prediction.squeeze(0).cpu().tolist()

    # Return the result as JSON
    return jsonify({"path": predicted_path_sequence})

# ---------------------------
# 4. Run the Server
# ---------------------------
if __name__ == "__main__":
    # This makes the server accessible on your local machine at http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=False)