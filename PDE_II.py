# Author: Mehak Aggarwal\
# Last Modified: Nov 10, 2020

# Partial Differential Equation\
# Equation: x' + t' - 3x - t = 0\
# Initial Condition: f(x=0, t) = 1, 2, 3, 4

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

x = torch.FloatTensor(n, 2).uniform_(-1, 1)
const = torch.round(torch.FloatTensor(n, 1).uniform_(0, 1) * const_range)

vol = torch.cat((x, const), dim=1)
surface = (vol).clone() * torch.Tensor([0, 1, 1])

x_train = torch.cat([vol, surface], dim=0)
x_train

# +
# testing data
n_test = int(n / 5)
const_val = 3

x = torch.FloatTensor(n_test, 2).uniform_(-1, 1)
const = torch.Tensor([[const_val]] * n_test)
print(const.shape)

vol_test = torch.cat((x, const), dim=1)
surface_test = (vol_test).clone() * torch.Tensor([0, 1, 1])

x_test = torch.cat([vol_test, surface_test], dim=0)
x, t = x_test[:, 0].numpy(), x_test[:, 1].numpy()
y_true = (x * x + t * x + const_val)

x_test
# -

f = nn.Sequential(nn.Linear(3, 80),
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
                                grad_outputs=torch.ones(fx.shape)) # [n * 2, 3]
    l_eq = dfdx[:n, 0] + dfdx[:n, 1] - (3 * x[:n, 0]) - x[:n, 1]
    l_bc = fx[n:, 0] - x[n:, 2] # Loss at boundary
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
        ax.set_ylabel('t')
        ax.set_zlim3d([round(y_true.min() - 1), round(y_true.max() + 1)])
        ax.set_zlabel('f(x,t)')

        ax.scatter3D(x, t, y_pred)
        ax.scatter3D(x, t, y_true)
        
        plt.savefig(f'results/surface{int(i / interval)}.png', dpi=100)
        plt.close()

# !convert -delay 100 results/surface*.png results/animated_surface.gif
from IPython.display import HTML
HTML('<img src="results/animated_surface.gif">')

from IPython.display import HTML
HTML('<img src="results/animated_surface.gif">')


