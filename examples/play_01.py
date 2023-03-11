import torch

f = torch.nn.Linear(10, 5)
# f.requires_grad_(False)
# x = torch.nn.Parameter(torch.rand(10), requires_grad=True)
x = torch.rand(10)
x.requires_grad_(True)
optim = torch.optim.SGD([x], lr=1e-1)
mse = torch.nn.MSELoss()
y = torch.ones(5)  # the desired network response

num_steps = 100  # how many optim steps to take

for i in range(num_steps):
   optim.zero_grad()
   loss = mse(f(x), y)
   loss.backward()
   optim.step()
   print(x)