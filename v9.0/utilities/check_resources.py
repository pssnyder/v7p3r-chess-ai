import torch
print('CUDA is available!' if torch.cuda.is_available() else 'Still on CPU :(')
print(f'Using: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')