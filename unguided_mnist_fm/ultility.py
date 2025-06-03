import torch
from typing import Optional, List, Type, Tuple, Dict
from gaussian import Density
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from torchvision.utils import make_grid, np
from ode import Simulator
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_conditional_path(
    path, num_rows: int = 3, num_cols: int = 3, num_timesteps: int = 12
):
    # Sample
    num_samples = num_rows * num_cols
    z, _ = path.p_data.sample(num_samples)
    z = z.view(-1, 1, 32, 32)

    # Setup plot
    fig, axes = plt.subplots(
        1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows)
    )

    # Sample from conditional probability paths and graph
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for tidx, t in enumerate(ts):
        tt = t.view(1, 1, 1, 1).expand(num_samples, 1, 1, 1)  # (num_samples, 1, 1, 1)
        xt = path.sample_conditional_path(z, tt)  # (num_samples, 1, 32, 32)
        grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1, 1))
        axes[tidx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[tidx].axis("off")

    plt.show()


def plot_training_loss(losses: List):
    plt.plot(np.arange(len(losses)), losses, label="Loss vs. # of epochs", color="red")

    # Customize the plot
    plt.xlabel("Parameter Value")
    plt.ylabel("Loss")
    plt.title("Loss Function Scatter Plot")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def plot_generated_sample(
    path,
    simulator: Simulator,
    num_rows: int = 5,
    num_cols: int = 5,
    num_timesteps: int = 1000,
):
    num_samples = num_rows * num_cols

    # Setup plot
    fig, ax = plt.subplots(1, 1, figsize=(2 * num_cols, 2 * num_rows))

    x0 = path.p_init.sample(num_samples)
    # Sample from conditional probability paths and graph
    ts = (
        torch.linspace(0, 1, num_timesteps)
        .view(1, -1, 1, 1, 1)
        .expand(num_samples, -1, 1, 1, 1)
        .to(device)
    )
    x1 = simulator.simulate(x0, ts)  # (num_samples, 1, 32, 32)
    grid = make_grid(x1, nrow=num_cols, normalize=True, value_range=(-1, 1))
    ax.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    ax.axis("off")

    plt.show()
