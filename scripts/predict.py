import warnings
import argparse
import h5py

import torch

import _commons

import wlmmuq.data as wlds
import wlmmuq.models as wlcnn

from wlmmuq import OFFSET
from wlmmuq.data import SCALE, NUM_WORKERS

MOMENT_ORDER = 1
IMGSIZE = 304
NIMGS = 72000
BATCH_SIZE = 256
IDX_DATASET = 'kappa_pred'

MODEL_CLASSES = {
    "tensorflow.UNet": (wlcnn.tensorflow.UNet, False),
    "tensorflow.UNetScoreMatching": (wlcnn.tensorflow.UNetScoreMatching, True),
    "torch.DRUNet": (wlcnn.torch.DRUNet, True),
    "torch.SUNet": (wlcnn.torch.SUNet, False)
} # (model_class, scale_as_input)

def main(
        path_to_trained_model, path_to_augmented_dataset, path_to_output_dataset,
        cosmos_include_faint=False, backend=None, arch=None, denoiser=False,
        use_stdnoise_mask=False, moment_order=MOMENT_ORDER, path_to_pred_dataset=None,
        imgsize=IMGSIZE, nimgs=NIMGS, batch_size=BATCH_SIZE, offset=OFFSET,
        idx_dataset=IDX_DATASET, seed=None, verbose=False, **kwargs
):
    _commons.set_seed(seed)

    keys_model = [
        'meancentering', 'sigmoid_activation', 'small_model'
    ]
    kwargs_model = {k: kwargs.pop(k) for k in keys_model if k in kwargs}
    try:
        no_bias = kwargs.pop("no_bias")
    except KeyError:
        pass
    else:
        kwargs_model.update(use_bias=not no_bias)

    if use_stdnoise_mask:
        std_noise, mask = _commons.get_stdnoise_mask(
            imgsize, cosmos_include_faint=cosmos_include_faint,
            convert_to_torch_tensor=True, seed=seed, verbose=verbose
        )
        kwargs.update(std_noise=std_noise, mask=mask)

    if arch is not None:
        backend = arch.split(".")[0]
        cnn_class, scale_as_input = MODEL_CLASSES[arch]
        if scale_as_input:
            kwargs.update(scale_as_input=scale_as_input)
    else:
        cnn_class = None
        scale_as_input = False

    if backend == 'tensorflow': # Use Keras (TensorFlow backend)
        data_module = wlds.tensorflow
        model_module = wlcnn.tensorflow
    elif backend == 'torch': # Use DeepInverse (PyTorch backend)
        data_module = wlds.torch
        model_module = wlcnn.torch
    else:
        raise ValueError

    if denoiser:
        dataset_class = data_module.HDF5DatasetDenoiser
    else:
        dataset_class = data_module.HDF5DatasetDeepMass

    if verbose:
        print("Initialize dataset")

    # *** CAUTION ***
    # Keyword arguments `sort_by_filename_ori` and `shuffle` must be set to
    # False in order input convergence maps `kappa_inp` to be stored in the
    # same order as the targets `kappa_true`.
    dataset = dataset_class(
        order=moment_order, hdf5_filepath=path_to_augmented_dataset,
        pred_filepath=path_to_pred_dataset,
        nimgs=nimgs, batch_size=batch_size,
        sort_by_filename_ori=False, shuffle=False,
        output_shape=imgsize,
        offset=offset, newaxis=True, **kwargs
    )
    kwargs_dataloader = {}
    if backend == 'tensorflow':
        kwargs_dataloader.update(raise_stop_iteration=True)
    dataloader = dataset.to_dataloader(**kwargs_dataloader)
    dataloader = iter(dataloader)

    # Initialize model
    if backend == 'tensorflow':
        model = model_module.load_model(
            path_to_trained_model, **kwargs_model
        )
    elif backend == 'torch':
        model = cnn_class(
            map_size=imgsize, offset=offset, **kwargs_model
        )
        checkpoint = torch.load(path_to_trained_model)
        model.load_state_dict(checkpoint['state_dict'])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Device: {device}")
        model.to(device)
    else:
        raise ValueError

    if verbose:
        model_module.print_model(model)

    with h5py.File(path_to_output_dataset, 'w') as file:
        try:
            del file[idx_dataset]
        except KeyError:
            pass
        else:
            warnings.warn(
                f"Found existing dataset for {idx_dataset}; "
                "it will be overwritten."
            )
        file.create_dataset(
            idx_dataset, shape=(nimgs, imgsize, imgsize),
            dtype='float32'
        )
        beg_idx = 0
        while True:
            try:
                kappa_1, kappa_2 = next(dataloader)
            except StopIteration:
                break
            end_idx = beg_idx + kappa_1.shape[0]
            print(f"Processing images {beg_idx} to {end_idx}")

            if backend == 'tensorflow':
                kappa_inp = kappa_1 # kappa_inp, kappa_true
                kappa_pred = model.predict(kappa_inp) # Shape = (nimgs, nx, ny, 1)
                kappa_pred = kappa_pred[..., 0] # Remove channel dimension
            elif backend == 'torch':
                kappa_inp = kappa_2 # kappa_true, kappa_inp
                kappa_inp = kappa_inp.to(device)
                with torch.no_grad():
                    kappa_pred = model(kappa_inp) # Shape = (nimgs, 1, nx, ny)
                    kappa_pred = kappa_pred.squeeze(-3) # Remove channel dimension
                    kappa_pred = kappa_pred.cpu().numpy()
            else:
                raise ValueError

            file[idx_dataset][beg_idx:end_idx] = kappa_pred
            beg_idx = end_idx

    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_trained_model", type=str,
        help="Path to the trained model (keras file)"
    )
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "path_to_output_dataset", type=str,
        help="Path to the output dataset to be created (HDF5 file)"
    )
    parser.add_argument(
        "--backend", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Deep learning framework used to train the model ('tensorflow' or 'torch'). "
            "Required if `--arch` is not provided. Default = None"
        )
    )
    parser.add_argument(
        "-a", "--arch", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the model. Possible values are: "
            f"{' | '.join(MODEL_CLASSES.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "-s", "--small-model", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Whether to use a small model. Only available for PyTorch models."
        )
    )
    parser.add_argument(
        "-d", "--denoiser", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Reconstruct the original convergence map from an input corrupted "
            "by a white Gaussian noise."
        )
    )
    parser.add_argument(
        "--use-std-noise", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Whether to apply a heteroscedastic noise to the input images "
            "(denoiser only)."
        )
    )
    parser.add_argument(
        "--scale", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Multiplicative factor for the noise level if the flag `--use-std-noise` "
            "if used, or noise standard deviation otherwise. If `scale_min` is provided, "
            "upper bound of the uniform distribution over which the actual scale is drawn. "
            f"Only useful if the flag `--denoiser` is used. Default = {SCALE}"
        )
    )
    parser.add_argument(
        "--scale-min", type=float,
        default=argparse.SUPPRESS,
        help=(
            "If provided, then the scale for the noise standard deviation will be drawn "
            "uniformly between `scale_min` and `scale` for each input image."
        )
    )
    parser.add_argument(
        "--input-method", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Weak lensing method used as input ('ks', 'wiener' or 'wiener_pgd'). "
            "Only used if option `--denoiser` is not activated. Default = None"
        )
    )
    parser.add_argument(
        "--moment-order", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Order of the moment network. "
            f"Default = {MOMENT_ORDER}"
        )
    )
    parser.add_argument(
        "--path-to-pred-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the prediction dataset (HDF5 file), computed with "
            "a previously-trained network. This is useful to train a moment "
            "network of order 2. Default = None"
        )
    )
    parser.add_argument(
        "--imgsize", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of pixels (width) in input images. "
            f"Default = {IMGSIZE}"
        )
    )
    parser.add_argument(
        "--nimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images in the dataset. "
            f"Default = {NIMGS}"
        )
    )
    parser.add_argument(
        "--meancentering", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Apply a meancentering operator at the output of the network."
        )
    )
    parser.add_argument(
        "--no-bias", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Do not use bias in convolution or batch "
            "normalization layers."
        )
    )
    parser.add_argument(
        "--sigmoid-activation", action='store_true',
        default=argparse.SUPPRESS,
        help="Use sigmoid activation function in the output layer."
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size for training and validation. "
            f"Default = {BATCH_SIZE}"
        )
    )
    parser.add_argument(
        "-w", "--num-workers", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of workers for parallel processing. Only work for PyTorch datasets. "
            f"Default = {NUM_WORKERS}"
        )
    )
    parser.add_argument(
        "--offset", type=float,
        default=argparse.SUPPRESS,
        help=(
            f"Default convergence value for a perfectly uniform universe. Default = {OFFSET:.2f}"
        )
    )
    parser.add_argument(
        "--seed", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Seed for the random number generators"
        )
    )
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        default=argparse.SUPPRESS
    )
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
