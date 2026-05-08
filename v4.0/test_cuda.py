import torch

print('=' * 60)
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('cuDNN Version:', torch.backends.cudnn.version())
    print('GPU Count:', torch.cuda.device_count())
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('GPU Compute Capability:', f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')
    print('=' * 60)
    print('✅ CUDA is ready for training!')
else:
    print('❌ CUDA not available - training will use CPU')
print('=' * 60)
