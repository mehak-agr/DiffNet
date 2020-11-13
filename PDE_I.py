# Author: Mehak Aggarwal\
# Last Modified: Nov 10, 2020

# Ordinary Differential Equation\
# Equation: y' + 3xy = 0\
# Initial Condition: y(x=0) = 1, 2, 3, 4

from IPython.display import HTML
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as p3
plt.style.use('seaborn-pastel')

epochs = 750
interval = 20

# +
# Training Data
n = 50000
const_range = 5

x = torch.FloatTensor(n, 1).uniform_(-1, 1)
const = torch.round(torch.FloatTensor(n, 1).uniform_(0, 1) * const_range)

line = torch.cat((x, const), dim=1)
edge = (line).clone() * torch.Tensor([0, 1])

x_train = torch.cat([line, edge], dim=0)
x_train

# +
# testing data
n_test = int(n / 5)
const_val = 3

x = torch.FloatTensor(n_test, 1).uniform_(-1, 1)
const = torch.Tensor([[const_val]] * n_test)
print(const.shape)

line_test = torch.cat((x, const), dim=1)
edge_test = (line_test).clone() * torch.Tensor([0, 1])

x_test = torch.cat([line_test, edge_test], dim=0)
x = x_test[:, 0].numpy()
y_true = (np.exp(x ** 2) + const_val)

x_test
# -

f = nn.Sequential(nn.Linear(2, 80),
                  nn.Softplus(),
                  nn.Linear(80, 1))
f.train()
optimizer = torch.optim.SGD(f.parameters(), lr = 5e-3, momentum = 0.99)


def loss(f, x):
    """
    Compute PDE loss for given pytorch module f (the neural network)
    and training data x.
    Args:
        f: pytroch module used to predict PDE solution
        x: (n, 3) tensor containing the training data
    Returns:
        torch variable representing the loss
    """

    n = x.size()[0] // 2
    x.requires_grad = True # [n * 2, 3]
    fx = f(x) # [n * 2, 1]
    dfdx, = torch.autograd.grad(fx,
                                x,
                                create_graph=True,
                                retain_graph=True,
                                grad_outputs=torch.ones(fx.shape)) # [n * 2, 2]
    l_eq = dfdx[:n, 0] + (3 * x[:n, 0] * fx[:n, 0])
    l_bc = fx[n:, 0] - x[n:, 1] # Loss at boundary
    return (l_eq ** 2).mean() + (l_bc ** 2).mean()


def update(i):
    optimizer.zero_grad()
    l = loss(f, x_train)
    l.backward()
    optimizer.step()
    print("Epoch {}: ".format(i), l.item(), end='\r')

    return f(x_test).detach().squeeze().numpy()


for i in range(epochs):
    y_pred = update(i)
    
    if (i) % interval == 0:
        fig = plt.figure(figsize=(10, 10))
        ax = p3.Axes3D(fig)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')

        ax.plot(x, y_pred)
        ax.plot(x, y_true)
        
        plt.savefig(f'results/line{int(i / interval)}.png', dpi=100)
        plt.close()

# !convert -delay 100 results/line*.png results/animated_line.gif
from IPython.display import HTML
HTML('<img src="results/animated_line.gif">')


