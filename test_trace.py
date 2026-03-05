import torch
class MyModule(torch.nn.Module):
    def forward(self, x, y, z):
        return x + y + z

m = MyModule()
traced = torch.jit.trace(m, (torch.zeros(1), torch.zeros(1), torch.zeros(1)))
print(traced.forward.schema)
