import torch
import torch.nn.functional as F

a = torch.randn([1, 1, 960, 540])

print(a.shape)

b = F.interpolate(a, size=(1920, 1080))
print(b.shape)