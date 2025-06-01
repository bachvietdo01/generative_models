import torch
from typing import Optional, List, Type, Tuple, Dict
from gaussian import Density
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )


def imshow_density(
    density: Density,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    bins: int,
    ax: Optional[Axes] = None,
    x_offset: float = 0.0,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(
        density.cpu(), extent=[x_min, x_max, y_min, y_max], origin="lower", **kwargs
    )


def contour_density(
    density: Density,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    bins: int,
    ax: Optional[Axes] = None,
    x_offset: float = 0.0,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.contour(
        density.cpu(), extent=[x_min, x_max, y_min, y_max], origin="lower", **kwargs
    )


def plot_comparison_heatmap(p_init: Density, p_data: Density, scale: float = 1.0):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    bins = 200

    x_bounds = [-scale, scale]
    y_bounds = [-scale, scale]

    axes[0].set_title("Heatmap of p_init")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    imshow_density(
        density=p_init,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        ax=axes[0],
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
    )

    axes[1].set_title("Heatmap of p_data")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    imshow_density(
        density=p_data,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        ax=axes[1],
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
    )

    axes[2].set_title("Heatmap of p_init and p_data")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    imshow_density(
        density=p_init,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
    )
    imshow_density(
        density=p_data,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
    )


def plot_conditional_path(
    path,
    p_init: Density,
    p_data: Density,
    scale: float = 1.0,
    num_samples: int = 1000,
):
    x_bounds = [-scale, scale]
    y_bounds = [-scale, scale]

    plt.figure(figsize=(10, 10))
    plt.xlim(*x_bounds)
    plt.ylim(*y_bounds)
    plt.title("Gaussian Conditional Probability Path")

    # Plot source and target
    imshow_density(
        density=p_init,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
    )
    imshow_density(
        density=p_data,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
    )

    # Sample conditioning variable z
    z = path.sample_conditioning_variable(1)
    ts = torch.linspace(0.0, 1.0, 7).to(device)

    # Plot z
    plt.scatter(z[:, 0].cpu(), z[:, 1].cpu(), marker="*", color="red", s=75, label="z")
    plt.xticks([])
    plt.yticks([])

    # Plot conditional probability path at each intermediate t
    for t in ts:
        zz = z.expand(num_samples, 2)
        tt = t.unsqueeze(0).expand(num_samples, 1)  # (samples, 1)
        samples = path.sample_conditional_path(zz, tt)  # (samples, 2)
        plt.scatter(
            samples[:, 0].cpu(),
            samples[:, 1].cpu(),
            alpha=0.25,
            s=8,
            label=f"t={t.item():.1f}",
        )

    plt.legend(prop={"size": 18}, markerscale=3)
    plt.show()


def plot_generated_sample(
    xts: torch.Tensor,
    ts: torch.Tensor,
    p_init: Density,
    p_data: Density,
    num_samples: int = 1000,
    num_timesteps: int = 300,
    num_marginals: float = 3.0,
    scale: float = 1.0,
    legend_size: int = 24,
    markerscale: float = 1.8,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    x_bounds = [-scale, scale]
    y_bounds = [-scale, scale]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Samples from Learned Marginal ODE", fontsize=20)

    # Plot source and target
    imshow_density(
        density=p_init,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        ax=ax,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
    )
    imshow_density(
        density=p_data,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        ax=ax,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
    )

    # Extract every n-th integration step to plot
    every_n = record_every(
        num_timesteps=num_timesteps, record_every=num_timesteps // num_marginals
    )
    xts_every_n = xts[:, every_n, :]  # (bs, nts // n, dim)
    ts_every_n = ts[0, every_n]  # (nts // n,)
    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].item()
        ax.scatter(
            xts_every_n[:, plot_idx, 0].detach().cpu(),
            xts_every_n[:, plot_idx, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={tt:.2f}",
        )

    ax.legend(prop={"size": legend_size}, loc="upper right", markerscale=markerscale)

    ax = axes[1]
    ax.set_title("Trajectories of Learned Marginal ODE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot source and target
    imshow_density(
        density=p_init,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        ax=ax,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
    )
    imshow_density(
        density=p_data,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        bins=200,
        ax=ax,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
    )

    for traj_idx in range(num_samples // 10):
        ax.plot(
            xts[traj_idx, :, 0].detach().cpu(),
            xts[traj_idx, :, 1].detach().cpu(),
            alpha=0.5,
            color="black",
        )

    plt.grid(True)
    plt.show()
