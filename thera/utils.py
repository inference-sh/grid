from functools import partial

import jax
import numpy as np


def repeat_vmap(fun, in_axes=[0]):
    for axes in in_axes:
        fun = jax.vmap(fun, in_axes=axes)
    return fun


def make_grid(patch_size: int | tuple[int, int]):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    offset_h, offset_w = 1 / (2 * np.array(patch_size))
    space_h = np.linspace(-0.5 + offset_h, 0.5 - offset_h, patch_size[0])
    space_w = np.linspace(-0.5 + offset_w, 0.5 - offset_w, patch_size[1])
    return np.stack(np.meshgrid(space_h, space_w, indexing='ij'), axis=-1)  # [h, w]


def interpolate_grid(coords, grid, order=0):
    """
    args:
        coords: Tensor of shape (B, H, W, 2) with coordinates in [-0.5, 0.5]
        grid: Tensor of shape (B, H', W', C)
    returns:
        Tensor of shape (B, H, W, C) with interpolated values
    """
    # convert [-0.5, 0.5] -> [0, size], where pixel centers are expected at
    # [-0.5 + 1 / (2*size), ..., 0.5 - 1 / (2*size)]
    coords = coords.transpose((0, 3, 1, 2))
    coords = coords.at[:, 0].set(coords[:, 0] * grid.shape[-3] + (grid.shape[-3] - 1) / 2)
    coords = coords.at[:, 1].set(coords[:, 1] * grid.shape[-2] + (grid.shape[-2] - 1) / 2)
    map_coordinates = partial(jax.scipy.ndimage.map_coordinates, order=order, mode='nearest')
    return jax.vmap(jax.vmap(map_coordinates, in_axes=(2, None), out_axes=2))(grid, coords)
