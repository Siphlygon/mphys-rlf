import numpy as np
from tqdm import tqdm
import torch
import numpy as np
from tqdm import tqdm

import utils.logging

logger = utils.logging.get_logger(__name__)


@torch.no_grad()
def edm_sampling(
    model,
    context_batch=None,
    label_batch=None,
    latents=None,
    *,
    image_size=80,
    batch_size=16,
    timesteps=25,
    guidance_strength=0.1,
    sigma_min=2e-3,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=torch.inf,
    S_noise=1,
):
    """
    Perform deterministic or stochastic sampling from EDM paper (arXiv:2206.00364).
    Setting S_churn = 0 results in deterministic sampling.

    Parameters
    ----------
    model : torch.nn.Module
        The energy-based model.
    context_batch : array_like, optional
        The context batch. Defaults to None.
    label_batch : array_like, optional
        The label batch. Defaults to None.
    latents : array_like, optional
        The seed gaussian noise images. Defaults to None.
    image_size : int, optional
        The size of the image. Defaults to 80.
    batch_size : int, optional
        The batch size. Defaults to 16.
    timesteps : int, optional
        The number of sampling timesteps. Defaults to 25.
    guidance_strength : numeric, optional
        The guidance strength 'omega' parameter. Can be any numeric type. Defaults to 0.1.
    sigma_min : numeric, optional
        The minimum noise level. Can be any numeric type. Defaults to 2e-3.
    sigma_max : numeric, optional
        The maximum noise level. Can be any numeric type. Defaults to 80.
    rho : numeric, optional
        The value of rho. Defaults to 7.
    S_churn : numeric, optional
        The value of S_churn. Defaults to 0.
    S_min : numeric, optional
        The minimum value of S. Can be any numeric type. Defaults to 0.
    S_max : numeric, optional
        The maximum value of S. Can be any numeric type. Defaults to torch.inf.
    S_noise : numeric, optional
        The value of S_noise. Defaults to 1.

    Returns
    -------
    list
        A list of image batches, where every entry correponds to one time step. Batches have shape (batch_size, 1, image_size, image_size).

    """

    # Set device
    device = next(model.parameters()).device
    logger.info(f"Sampling on device: {device}")

    # If passed, prepare latents
    if latents is not None:
        assert latents.shape[1] == 1, "Latents must have 1 channel."
        assert (
            latents.shape[2] == image_size
        ), f"Latents must have size {image_size}x{image_size}."
        batch_size = latents.shape[0]
        # Make sure it's torch tensor
        if type(latents) != torch.Tensor:
            latents = torch.tensor(latents, device=device, dtype=torch.float32)

    # If not passed, sample latents from normal distribution.
    # Scaling will happen before the first sampling loop.
    else:
        latents = torch.randn([batch_size, 1, image_size, image_size], device=device)

    # Get noise level limits from model, which may be more restrictive
    model_sigma_min = (
        model.module.sigma_min
        if isinstance(model, torch.nn.DataParallel)
        else model.sigma_min
    )
    model_sigma_max = (
        model.module.sigma_max
        if isinstance(model, torch.nn.DataParallel)
        else model.sigma_max
    )

    # Update noise level limits
    sigma_min = max(sigma_min, model_sigma_min)
    sigma_max = min(sigma_max, model_sigma_max)

    # Generate time steps (= noise levels).
    sigma_steps = get_sampling_noise_levels(
        timesteps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
    )

    # If passed, prepare context and labels
    if context_batch is not None:
        assert context_batch.shape[0] == batch_size, (
            f"Context batch size ({context_batch.shape[0]})"
            f"must match batch size ({batch_size})."
        )
        if type(context_batch) != torch.Tensor:
            context_batch = torch.tensor(context_batch)

    if label_batch is not None:
        assert label_batch.shape[0] == batch_size, (
            f"Label batch size ({label_batch.shape[0]})"
            f"must match batch size ({batch_size})."
        )
        if type(label_batch) != torch.Tensor:
            label_batch = torch.tensor(label_batch)

    # Move all tensors to gpu
    sigma_steps = sigma_steps.to(device)
    if context_batch is not None:
        context_batch = context_batch.to(device)
    if label_batch is not None:
        label_batch = label_batch.to(device)

    # Prepare sampling loop.
    imgs = []
    x_next = latents * sigma_steps[0]  # Generate initial sample at t_0
    imgs.append(x_next.cpu())

    # Sampling loop:
    for i, (sigma_cur, sigma_next) in tqdm(
        enumerate(zip(sigma_steps[:-1], sigma_steps[1:])),
        desc="Sampling...",
        total=timesteps,
    ):
        # Update current image (= output from previous iteration)
        x_cur = x_next

        # Stochastic sampling: Increase noise temporarily
        if S_churn > 0:
            sigma_cur, x_cur = stochastic_churn(
                timesteps, S_churn, S_min, S_max, S_noise, sigma_cur, x_cur
            )

        # Calculate denoised image with forward model pass
        denoised = denoised_guided(
            model,
            x_cur,
            sigma_cur,
            context=context_batch,
            class_labels=label_batch,
            guidance_strength=guidance_strength,
        )

        # Score estimate
        d_cur = (x_cur - denoised) / sigma_cur

        # Euler step
        x_next = x_cur + d_cur * (sigma_next - sigma_cur)

        # Apply 2nd order correction
        if i < timesteps - 1:

            # Denoised image for next step
            denoised = denoised_guided(
                model,
                x_next,
                sigma_next,
                context=context_batch,
                class_labels=label_batch,
                guidance_strength=guidance_strength,
            )

            # Score estimate for next step
            d_next = (x_next - denoised) / sigma_next

            # 2nd order correction by applying trapezoidal rule
            x_next = x_cur + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_next)

        # Append to list
        imgs.append(x_next.cpu())

    return imgs


def get_sampling_noise_levels(timesteps, sigma_min=2e-3, sigma_max=80, rho=7):
    """
    Generate noise levels for each sampling step, according to the scheme proposed
    in the EDM paper (arXiv:2206.00364).

    Parameters
    ----------
    timesteps : int
        The number of sampling steps.
    sigma_min : numeric, optional
        The minimum noise level. Defaults to 2e-3.
    sigma_max : numeric, optional
        The maximum noise level. Defaults to 80.
    rho : numeric, optional
        The value of rho. Defaults to 7.

    Returns
    -------
    tuple
        A tuple containing the time step indices and the noise levels.
    """
    # Time step indices
    step_inds = torch.arange(timesteps)

    # Noise level for each time step
    rho_inv = 1 / rho
    sigma_steps = (
        sigma_max**rho_inv
        + step_inds / (timesteps - 1) * (sigma_min**rho_inv - sigma_max**rho_inv)
    ) ** rho

    # Add t_N=0 at the end
    sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])

    return sigma_steps


@torch.no_grad()
def denoised_guided(
    model,
    img,
    sigma,
    context=None,
    class_labels=None,
    guidance_strength=0.1,
):
    """
    Calculate the denoised image. Guidance is applied if context or class labels are passed.

    Parameters
    ----------
    model : torch.nn.Module
        The energy-based model.
    img : torch.Tensor
        The input image.
    sigma : torch.Tensor
        The noise level.
    context : torch.Tensor, optional
        The context tensor. Defaults to None.
    class_labels : torch.Tensor, optional
        The class labels. Defaults to None.
    guidance_strength : numeric, optional
        The guidance strength 'omega' parameter. Can be any numeric type. Defaults to 0.1.

    Returns
    -------
    torch.Tensor
        The denoised image.
    """
    # Set batch size
    batch_size = img.shape[0]
    sigma = sigma.expand(batch_size)

    # Calculate denoised image with forward model pass
    denoised = model(img, sigma, context=None)

    # Apply class conditioning if labels are provided
    if (class_labels is not None or context is not None) and guidance_strength:

        # Denoised image with class conditioning
        denoised_cond = model(
            img,
            sigma,
            class_labels=class_labels.long() if class_labels is not None else None,
            context=context,
        )

        # Apply guidance
        denoised = (
            1 + guidance_strength
        ) * denoised_cond - guidance_strength * denoised

    return denoised


def stochastic_churn(timesteps, S_churn, S_min, S_max, S_noise, sigma_cur, x_cur):
    """
    Add stochastic churn for stochastic sampling, i.e. temporarily increase noise level.

    Parameters
    ----------
    timesteps : int
        The number of timesteps.
    S_churn : int
        The value of S_churn.
    S_min : int
        The minimum value of S.
    S_max : torch.Tensor
        The maximum value of S.
    S_noise : int
        The value of S_noise.
    sigma_cur : torch.Tensor
        The current noise level.
    x_cur : torch.Tensor
        The current image.

    Returns
    -------
    tuple
        A tuple containing the updated noise level and image.
    """
    # Factor by which noise level is increased (gamma = 0 if S_churn = 0)
    gamma = (
        min(S_churn / timesteps, np.sqrt(2) - 1)
        if S_min <= sigma_cur.item() <= S_max
        else 0
    )

    # Increase noise level (sigma_hat = sigma_cur if gamma = 0).
    sigma_hat = (1 + gamma) * sigma_cur

    return sigma_hat, x_cur
