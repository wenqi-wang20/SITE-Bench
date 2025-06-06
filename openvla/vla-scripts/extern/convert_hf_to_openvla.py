import torch
from safetensors.torch import load_file
import glob
# List of your .safetensors file paths
# safetensors_files = [
#     'model_part1.safetensors',
#     'model_part2.safetensors',
#     # Add more files as needed
# ]

MODEL_PATH = '/dataext4/users/baolinpeng/hf_models/Meta-Llama-3.1-8B-Instruct'
# Initialize an empty dictionary to hold the combined weights
combined_weights = {}


# Load each .safetensors file and aggregate the parameters
for file in glob.glob(f'{MODEL_PATH}/*.safetensors'):
    # Load the weights from the .safetensors file
    weights = load_file(file)
    
    # Combine weights into the combined_weights dictionary
    for key, value in weights.items():
        # If the key already exists, you might want to sum or average the weights
        # depending on your use case. Here, we simply overwrite.
        # combined_weights[key.replace("language_model.", "llm.")] = value
        combined_weights[f"llm.{key}"] = value

# Create a new model and load the combined weights
# For example, let's assume you have a model class called MyModel
# model = MyModel()
# model.load_state_dict(combined_weights)

# Optionally, save the combined weights back to a .safetensors file
# from safetensors.torch import save_file
# save_file(combined_weights, 'combined_model.safetensors')

# import pdb
# pdb.set_trace()

combined_weights['llm.model.embed_tokens.weight'] = torch.cat((combined_weights['llm.model.embed_tokens.weight'], torch.randn(64,4096) * 0.0001), dim=0)
combined_weights['llm.lm_head.weight'] = torch.cat((combined_weights['llm.lm_head.weight'], torch.randn(64,4096) * 0.0001), dim=0)

combined_weights['llm.model.embed_tokens.weight'] = combined_weights['llm.model.embed_tokens.weight'].to(torch.bfloat16)
combined_weights['llm.lm_head.weight'] = combined_weights['llm.lm_head.weight'].to(torch.bfloat16)
# Rename dict map

torch.save(combined_weights, 'llm_backbone.pt')

model = torch.load('/dataext4/users/baolinpeng/hf_models/vla-models/custom-llama31-instruct+8b+siglip/checkpoints/latest-checkpoint.pt')

llm_backbone = torch.load('/home/baolinpeng/experiments/openvla/vla-scripts/extern/llm_backbone.pt')
model['model']['llm_backbone'] = llm_backbone
torch.save(model, '/dataext4/users/baolinpeng/hf_models/vla-models/custom-llama31-instruct+8b+siglip/checkpoints/latest-checkpoint.pt')
