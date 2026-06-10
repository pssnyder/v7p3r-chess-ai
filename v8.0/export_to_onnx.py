import torch
import os
from pathlib import Path
from src.network import V8ValueNetwork
from src.reward_shaper import RewardShaper

# 1. Instantiate the model classes
value_model = V8ValueNetwork(input_dim=55, dropout_rate=0.3)
shaper_model = RewardShaper(feature_dim=55, num_feature_groups=10)

# Determine which is the latest model from opponent training
opponent_model_path = "training/v8_opponent_training"

# Find the latest generation number by looking at value_network files
value_network_files = sorted([f for f in os.listdir(opponent_model_path) if f.endswith('_value_network.pt')])
if not value_network_files:
    raise FileNotFoundError(f"No value_network models found in {opponent_model_path}")

# Extract generation number from the latest file (e.g., gen_0018_value_network.pt)
latest_value_network = value_network_files[-1]
gen_number = latest_value_network.split('_')[1]  # Extract '0018' from 'gen_0018_value_network.pt'

# Construct paths for both models from the same generation
value_network_path = os.path.join(opponent_model_path, f"gen_{gen_number}_value_network.pt")
reward_shaper_path = os.path.join(opponent_model_path, f"gen_{gen_number}_reward_shaper.pt")

# Verify both files exist
if not os.path.exists(value_network_path):
    raise FileNotFoundError(f"Value network not found: {value_network_path}")
if not os.path.exists(reward_shaper_path):
    raise FileNotFoundError(f"Reward shaper not found: {reward_shaper_path}")

# 2. Load the models
print(f"Loading value network from: {value_network_path}")
value_model.load_state_dict(torch.load(value_network_path))
value_model.eval()

print(f"Loading reward shaper from: {reward_shaper_path}")
shaper_model.load_state_dict(torch.load(reward_shaper_path))
shaper_model.eval()

# 3. Create dummy input matching your input shape (batch_size, feature_dim)
dummy_input = torch.randn(1, 55)

# 4. Export value network to ONNX
value_onnx_path = f"gen_{gen_number}_value_network.onnx"
print(f"Exporting value network to: {value_onnx_path}")
torch.onnx.export(
    value_model, 
    dummy_input, 
    value_onnx_path, 
    input_names=['board_features'], 
    output_names=['state_value'],
    dynamic_axes={'board_features': {0: 'batch_size'}, 'state_value': {0: 'batch_size'}}
)

# 5. Export reward shaper to ONNX
shaper_onnx_path = f"gen_{gen_number}_reward_shaper.onnx"
print(f"Exporting reward shaper to: {shaper_onnx_path}")
torch.onnx.export(
    shaper_model, 
    dummy_input, 
    shaper_onnx_path, 
    input_names=['board_features'], 
    output_names=[f'feature_group_{i}' for i in range(10)],
    dynamic_axes={'board_features': {0: 'batch_size'}, **{f'feature_group_{i}': {0: 'batch_size'} for i in range(10)}}
)

print(f"\n✅ Successfully exported both models from generation {gen_number}")
print(f"   - Value Network: {value_onnx_path}")
print(f"   - Reward Shaper: {shaper_onnx_path}")