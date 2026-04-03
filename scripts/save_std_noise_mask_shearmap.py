import argparse
import torch

import wlmmuq

import _commons
import _add_arguments

def main(
        path_to_std_noise: str | None = wlmmuq.PATH_TO_STD_NOISE,
        path_to_mask: str | None = wlmmuq.PATH_TO_MASK,
        path_to_real_shearmap: str | None = wlmmuq.PATH_TO_REAL_SHEARMAP,
        cosmos_include_faint: bool = False,
        max_z: float | None = _commons.MAX_Z,
        use_zbins: bool = False,
        path_to_zbins: str | None = wlmmuq.PATH_TO_ZBINS,
        idx_zbins: list[int] = _commons.IDX_ZBINS,
        resolution: float = _commons.RESOLUTION,
        imgsize: int = _commons.IMGSIZE,
        seed: int | None = None, verbose: bool = False
):
    _commons.set_seed(seed)

    # Load noise standard deviation and mask
    if use_zbins:
        assert path_to_zbins is not None
        zbins = wlmmuq.utils.get_zbins(path_to_zbins, idx_zbins=idx_zbins)
    else:
        zbins = None

    std_noise, mask, gamma_real = _commons.get_stdnoise_mask_shearmap(
        bin_data_from_cosmos=True,
        get_noisy_shear_map=True,
        imgsize=imgsize, cosmos_include_faint=cosmos_include_faint,
        max_z=max_z, resolution=resolution,
        east_right=True, zbins=zbins,
        inpainting=False, verbose=verbose
    )

    assert path_to_std_noise is not None
    assert path_to_mask is not None
    assert path_to_real_shearmap is not None
    torch.save(std_noise, path_to_std_noise)
    torch.save(mask, path_to_mask)
    torch.save(gamma_real, path_to_real_shearmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _add_arguments.path_to_std_noise_mask_gamma(parser)
    _add_arguments.cosmos_zbins(parser)
    _add_arguments.resolution(parser)
    _add_arguments.imgsize(parser)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
