import torch

x = torch.randn(1,3,3)
print(x)
y = x.argmax(dim=2)
# y = torch.argmax(x, dim=2)

print(y)
