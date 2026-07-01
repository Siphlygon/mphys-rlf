import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import numpy as np
import torch


def random_image_grid(dset, n_img=25, idx_titles=False, **kwargs):
    def has_context(img):
        return isinstance(img, list) or isinstance(img, tuple)

    idxs = np.random.choice(len(dset), n_img, replace=False)
    imgs = [img if not has_context(img := dset[i]) else img[0] for i in idxs]

    if idx_titles:
        kwargs["titles"] = idxs
    return plot_image_grid(imgs, **kwargs)


def plot_image_grid(
    imgs,
    suptitle=None,
    vmin=0,
    vmax=1,
    savefig=None,
    fig_axs=None,
    n_rows=None,
    n_cols=None,
    titles=None,
    **imshow_kwargs,
):
    if isinstance(imgs, list):
        imgs = np.array(imgs)

    if n_rows is None and n_cols is None:
        n = int(np.sqrt(imgs.shape[0]))
        n_cols = n
        n_rows = n + np.ceil((imgs.shape[0] - n**2) / n).astype(int)
    elif n_rows is None or n_cols is None:
        known = n_rows or n_cols
        n = int(imgs.shape[0] // known) + int(bool(imgs.shape[0] % known))
        n_rows = n_rows or n
        n_cols = n_cols or n

    fig, axs = fig_axs or plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        constrained_layout=True,
        figsize=(3 * n_cols, 3 * n_rows),
    )
    flat_axs = axs.flat if isinstance(axs, np.ndarray) else [axs]

    if titles is not None:
        assert len(titles) == len(imgs), (
            f"Number of titles ({len(titles)}) should match number of images "
            f"({len(imgs)})."
        )
    for i, (ax, img) in enumerate(zip(flat_axs, imgs)):
        ax.axis("off")
        ax.imshow(img.squeeze(), vmin=vmin, vmax=vmax, **imshow_kwargs)

        # Set axis title if titles are passed
        if titles is not None:
            title = titles[i]
            # Convert title to string if necessary
            match title:
                case float() | np.float32() | np.float64():
                    t_str = f"{title:.2e}"
                case int():
                    t_str = str(title)
                case _:
                    t_str = title

            ax.set_title(t_str)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize="xx-large")

    if savefig is not None:
        fig.savefig(savefig)

    return fig, axs


def plot_image_grid_from_file(path, save=False, **kwargs):
    imgs = torch.load(path, map_location="cpu").numpy()[:, -1, :, :]
    savefig = path.parent / f"{path.stem}_grid.png" if save else None
    return random_image_grid(imgs, savefig=savefig, **kwargs)
