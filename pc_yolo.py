from ultralytics.nn.tasks import *
import torch

input_tensor = torch.randn(1, 256, 64, 64)

# ## test split
# a1, a2 = input_tensor.split((128, 128), dim=1)
# print(f"{a1.shape}, {a2.shape}")


# ## test C3K2
# c3k2 = C3k2(c1=256, c2=256, n=3, c3k=True)
# output_tensor = c3k2(input_tensor)
# print(f"C3K2: {output_tensor.shape}")


# ## test C2PSA (Yolo11 attn)
# c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
# output_tensor = c2psa(input_tensor)
# print(f"C2PSA: {output_tensor.shape}")


## test A2C2f (Yolo12 flash attn)
a2c2f = A2C2f(256, 256, n=1, a2=True, area=1)
output_tensor3 = a2c2f(input_tensor)
print(f"A2C2f: {output_tensor3.shape}")