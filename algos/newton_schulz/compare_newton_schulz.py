import torch
import time
from newton_schulz_baseline import zeropower_via_newtonschulz5_baseline
from newton_schulz_triton import zeropower_via_newtonschulz5_triton

# Generate a random matrix G
N, M = 256, 256
G = torch.randn(N, M, device='cuda', dtype=torch.float32)

# Time the original Python implementation
start = time.time()
X_python = zeropower_via_newtonschulz5_baseline(G.clone(), steps=10)
torch.cuda.synchronize()
time_python = time.time() - start

# Time the Triton implementation
start = time.time()
X_triton = zeropower_via_newtonschulz5_triton(G.clone(), steps=10)
torch.cuda.synchronize()
time_triton = time.time() - start

# Compare the results
difference = (X_python - X_triton).abs().max()
print(f'Max difference between Python and Triton implementations: {difference.item()}')

# Check if the difference is within acceptable tolerance
tolerance = 1e-4
if difference < tolerance:
    print('Test passed!')
else:
    print('Test failed.')

print(f'Python implementation time: {time_python * 1000:.2f} ms')
print(f'Triton implementation time: {time_triton * 1000:.2f} ms')
