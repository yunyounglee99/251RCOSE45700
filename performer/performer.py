import torch
from performer_pytorch import Performer

model = Performer(
  dim = 512,
  depth = 1,
  heads = 8,
  causal = True
)

# example일 뿐입니다.
x = torch.randn(1, 2048, 512)
model(x)