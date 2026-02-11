import torch

import wlmmuq

import _commons

imgsize = 384

std_noise, mask = _commons.get_stdnoise_mask(
    path_to_std_noise=wlmmuq.PATH_TO_STD_NOISE,
    path_to_mask=wlmmuq.PATH_TO_MASK,
    imgsize=imgsize, cosmos_include_faint=False,
    inpainting=True, verbose=True
)
physics = wlmmuq.physics.MassMapping(sigma=std_noise, mask=mask)

kappa = torch.randn(2, 6, imgsize, imgsize)
gamma = physics(kappa)
