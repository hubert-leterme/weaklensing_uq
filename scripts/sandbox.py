import torch
import wlmmuq.transform as wltransf

NBINS = 6
NIMGS = 8
IMGSIZE = 16
WHICH_WAY = 1
TRANSPOSE = True

if __name__ == "__main__":

    dchi = torch.rand(NBINS)
    chi = torch.cumsum(dchi, dim=0)
    bnt = wltransf.BNT(chi)

    x = torch.randn(NIMGS, NBINS, IMGSIZE, IMGSIZE)
    y = bnt(x, which_way=WHICH_WAY, transpose=TRANSPOSE)
    x0 = bnt.inverse(y, which_way=WHICH_WAY, transpose=TRANSPOSE)

    nrmse = torch.linalg.norm(x0 - x) / torch.linalg.norm(x)
    print(f"NRMSE = {nrmse:.1e}")
