import torch
t = torch.empty((100, 100), pin_memory=True)
print(f"Address: {t.data_ptr()}, Modulo 512: {t.data_ptr() % 512}")
