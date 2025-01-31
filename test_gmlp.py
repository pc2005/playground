import torch
from torch import nn
from g_mlp import gMLP, gMLPVision

# model = gMLP(
#     num_tokens = 20000,
#     dim = 512,
#     depth = 6,
#     seq_len = 256,
#     attn_dim=64,
#     circulant_matrix = True,      # use circulant weight matrix for linear increase in parameters in respect to sequence length
#     act = nn.Tanh()               # activation for spatial gate (defaults to identity)
# )

# x = torch.randint(0, 20000, (1, 256))
# logits = model(x) # (1, 256, 20000)
# print(logits)