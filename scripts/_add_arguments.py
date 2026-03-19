import argparse
import wlmmuq as wl

from wlmmuq.datasets import NUM_WORKERS, OPENINGANGLE

import _commons

def create_dataset(parser, path_to_output, idx_lp):

    parser.add_argument(
        "-o", "--path-to-output", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the output HDF5 file. "
            f"Default = {path_to_output}"
        )
    )
    parser.add_argument(
        "--idx-lp", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Index of the learning potential, indicating which folder to look "
            "into for the HDF5 files containing the dataset (`LPxxx` where `xxx` "
            f"ranges from `001` to `100`). Default = {idx_lp}"
        )
    )
    parser.add_argument(
        "--openingangle", type=float,
        default=argparse.SUPPRESS,
        help=f"Opening angle (deg). Default = {OPENINGANGLE}"
    )
    parser.add_argument(
        "--ninpimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            f"Number of input images. Default = {_commons.NINPIMGS}"
        )
    )
    parser.add_argument(
        "-z", "--use-zbins", action='store_true',
        default=argparse.SUPPRESS,
        help=("Create tomographic kappa-maps with redshift bins.")
    )
    parser.add_argument(
        "--path-to-zbins", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the FITS file containing the values delimitating the redshift bins. "
            f"Default = {_commons.PATH_TO_ZBINS}"
        )
    )
    parser.add_argument(
        "--idx-zbins", type=int, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            "Select redshift bin indices. Used to group and merge redshift bins. "
            "By default, the redshift bins provided by Euclid are grouped two by two "
            "(except for the three first bins, which are grouped together), which "
            f"yields six bins in total. Default = {_commons.IDX_ZBINS}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size, to avoid memory overload. "
            "Default = 50"
        )
    )
    parser.add_argument(
        "--update-metadata-only", action='store_true',
        default=argparse.SUPPRESS,
    )


def _get_model_classes(denoiser=False, deepmass=False):
    
    if not deepmass:
        model_classes = wl.models.MODEL_CLASSES_DENOISER
    elif not denoiser:
        model_classes = wl.models.MODEL_CLASSES_DEEPMASS
    else:
        model_classes = wl.models.MODEL_CLASSES

    return model_classes


def model(parser, uq=False, denoiser=False, deepmass=False):

    model_classes = _get_model_classes(denoiser=denoiser, deepmass=deepmass)
    parser.add_argument(
        "-a", "--arch", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the model. Possible values are: "
            f"{' | '.join(model_classes.keys())}. Default = None"
        )
    )
    if deepmass:
        parser.add_argument(
            "-m", "--mode-preproc", type=str,
            default=argparse.SUPPRESS,
            help=(
                "Preprocessing mode for DeepMass: 'wiener' or 'ks'. "
                "Default = None"
            )
        )
    else:
        parser.add_argument(
            "-s", "--model-size", type=str,
            default=argparse.SUPPRESS,
            help=(
                "Size of the model (DRUNet only). Possible values are: "
                f"{' | '.join(wl.models.nn.MODEL_SIZE_DRUNET.keys())}. Default = None"
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
        "--model-specs", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Name of the subdirectory containing the saved checkpoints. "
            f"Default = '{_commons.MODEL_SPECS[0]}' for order-1 networks and "
            f"'{_commons.MODEL_SPECS[1]}' for order-2 networks."
        )
    )
    if uq:
        parser.add_argument(
            "--additional-outlayer", type=str,
            default=argparse.SUPPRESS,
            help=(
                "Type of additional output layer. "
                "Only used for training order-2 models. "
                "Possible values are: 'meancentering' | 'leakyrelu'. "
                "In any case, ReLU is applied at the output in evaluation mode. "
                "Default = None"
            )
        )


def model_order1(parser, denoiser=False, deepmass=False):

    model_classes = _get_model_classes(denoiser=denoiser, deepmass=deepmass)
    parser.add_argument(
        "-a1", "--arch-order1", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the order-1 model. Possible values are: "
            f"{' | '.join(model_classes.keys())}. Default = None"
        )
    )
    if deepmass:
        parser.add_argument(
            "-m1", "--mode-preproc-order1", type=str,
            default=argparse.SUPPRESS,
            help=(
                "Preprocessing mode for DeepMass (order-1 model): 'wiener' or 'ks'. "
                "Default = None"
            )
        )
    else:
        parser.add_argument(
            "-s1", "--model-size-order1", type=str,
            default=argparse.SUPPRESS,
            help=(
                "Size of the order-1 model (DRUNet only). Possible values are: "
                f"{' | '.join(wl.models.nn.MODEL_SIZE_DRUNET.keys())}. Default = None"
            )
        )
    parser.add_argument(
        "--no-bias-order1", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Do not use bias in convolution or batch "
            "normalization layers (order-1 model)."
        )
    )
    parser.add_argument(
        "--model-specs-order1", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Name of the subdirectory containing the saved checkpoints "
            f"for order-1 netowrks. Default = '{_commons.MODEL_SPECS[0]}'."
        )
    )


def model_uq(parser, denoiser=False, deepmass=False):

    model_classes = _get_model_classes(denoiser=denoiser, deepmass=deepmass)
    parser.add_argument(
        "-auq", "--arch-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the order-2 model, if different from `--arch`. "
            "Possible values are: "
            f"{' | '.join(model_classes.keys())}. Default = None"
        )
    )
    if deepmass:
        parser.add_argument(
            "-muq", "--mode-preproc-uq", type=str,
            default=argparse.SUPPRESS,
            help=(
                "Preprocessing mode for DeepMass (order-2 model): 'wiener' or 'ks'. "
                "Default = None"
            )
        )
    else:
        parser.add_argument(
            "-suq", "--model-size-uq", type=str,
            default=argparse.SUPPRESS,
            help=(
                "Size of the order-2 model (DRUNet only). Possible values are: "
                f"{' | '.join(wl.models.nn.MODEL_SIZE_DRUNET.keys())}. Default = None"
            )
        )
    parser.add_argument(
        "--no-bias-uq", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Do not use bias in convolution or batch "
            "normalization layers (order-2 models)."
        )
    )
    parser.add_argument(
        "--model-specs-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Name of the subdirectory containing the saved checkpoints "
            f"for order-2 netowrks. Default = '{_commons.MODEL_SPECS[1]}'."
        )
    )
    parser.add_argument(
        "--additional-outlayer-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Type of additional output layer (order-2 model). "
            "Possible values are: 'meancentering' | 'leakyrelu'. "
            "In any case, ReLU is applied at the output in evaluation mode. "
            "Default = None"
        )
    )


def model_name(parser):

    parser.add_argument(
        "-c", "--model-name", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Model name, used for setting the subdirectory containing "
            "the saved checkpoints. Default is None."
        )
    )


def checkpoint(parser):

    model_name(parser)
    parser.add_argument(
        "-t", "--timestamp", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Timestamp of the model to load. "
            "Default = None"
        )
    )
    parser.add_argument(
        "-e", "--epoch", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Epoch of the model to load. "
            f"Default = {_commons.EPOCH}"
        )
    )
    parser.add_argument(
        "-uq", "--load-model-uq", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Load the order-2 moment network, for UQ."
        )
    )
    parser.add_argument(
        "-c0", "--checkpoint-subdir-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Checkpoint subdirectory for the order-2 moment network, "
            "if different from `--checkpoint-subdir`."
        )
    )
    parser.add_argument(
        "-t0", "--timestamp_uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Timestamp of the order-1 model to load. "
            "Default = None"
        )
    )
    parser.add_argument(
        "-e0", "--epoch_uq", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Epoch of the model to load. "
            f"Default is the same value as `--epoch` ({_commons.EPOCH} if not provided)."
        )
    )


def cosmos(parser):
    parser.add_argument(
        "--max-z", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Maximum redshift value, above which the COSMOS catalog will be "
            f"filtered out. Default is {_commons.MAX_Z}"
        )
    )
    parser.add_argument(
        "--cosmos-include-faint", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Include the 'faint' COSMOS catalog in "
            "addition to the 'bright' one."
        )
    )


def std_noise_mask(parser):

    parser.add_argument(
        "--bin-data-from-cosmos", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Whether to compute the noise standard deviation, mask "
            "and noisy shear map from the COSMOS catalog."
        )
    )
    cosmos(parser)
    parser.add_argument(
        "--resolution", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Resolution (in arcmin per pixel) at which the COSMOS catalog will "
            f"be binned. Default = {_commons.RESOLUTION:.2f}"
        )
    )


def imgsize(parser):

    parser.add_argument(
        "--imgsize", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of pixels (width) in input images. "
            f"Default = {_commons.IMGSIZE}"
        )
    )


def dataset(parser, batch_size):

    imgsize(parser)
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size. "
            f"Default = {batch_size}"
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


def train_val_dataset(parser, batch_size):

    parser.add_argument(
        "--path-to-train-val-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the training and validation sets (HDF5 file). "
            f"Default = {wl.PATH_TO_TRAIN_VAL_DATASET}"
        )
    )
    parser.add_argument(
        "--nimgs-train", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of training examples. "
            f"Default = {_commons.NIMGS_TRAIN}"
        )
    )
    parser.add_argument(
        "--nimgs-val", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of validation examples. "
            f"Default = {_commons.NIMGS_CALIB}"
        )
    )
    dataset(parser, batch_size)


def test_calib_dataset(parser, batch_size):

    parser.add_argument(
        "--path-to-real-shearmap", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the COSMOS shear map (PyTorch saved object). "
            f"Default = {wlmmuq.PATH_TO_REAL_SHEARMAP}"
        )
    )
    parser.add_argument(
        "--path-to-test-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the test set (HDF5 file). "
            f"Default = {wl.PATH_TO_TEST_DATASET}"
        )
    )
    parser.add_argument(
        "--path-to-calib-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the calibration set (HDF5 file). "
            f"Default = {wl.PATH_TO_CALIB_DATASET}"
        )
    )
    parser.add_argument(
        "--test-on-real-data", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Whether to test the mass mapping method on the "
            "COSMOS shear map."
        )
    )
    parser.add_argument(
        "--run-both", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Run inference on both the simulated test set and the real COSMOS shear map. "
            "Calibration (if requested) is still performed only once."
        )
    )
    parser.add_argument(
        "--nimgs-test", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of test examples. "
            f"Default = {_commons.NIMGS_TEST}"
        )
    )
    parser.add_argument(
        "--nimgs-calib", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of calibration examples. "
            f"Default = {_commons.NIMGS_CALIB}"
        )
    )
    parser.add_argument(
        "-f", "--min-idx-filename-ori-calib",
        type=int, default=argparse.SUPPRESS,
        help=(
            "Filter images by filenames with indices equal or larger than this value. "
            f"Default = {_commons.MIN_IDX_FILENAME_ORI_CALIB}."
        )
    )
    dataset(parser, batch_size)


def cqr(parser, prompt_init_bounds=False, montecarlo=False, zero_init_bounds=False):

    parser.add_argument(
        "--cqr", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Calibrate with CQR."
        )
    )
    parser.add_argument(
        "--mode-cqr", type=str, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            f"Mode for CQR. Possible values are: {' | '.join(wl.models.CQR_CLASSES.keys())}. "
            f"Several values can be provided. Default = '{_commons.MODE_CQR}'"
        )
    )
    parser.add_argument(
        "--scaling-factor-chisqcqr", type=float, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            "Scaling factor for Chi-squared CQR. "
            "Only used if `--mode-cqr` is set to 'chisqcqr'. "
            "Several values can be provided. Default = None"
        )
    )
    parser.add_argument(
        "--confidence-uq", type=float,
        default=argparse.SUPPRESS,
        help=f"Level of confidence for UQ. Default = {_commons.CONFIDENCE_UQ:.1f}-sigma"
    )
    if prompt_init_bounds:
        if not montecarlo:
            uq_method = "using an analytical method"
        else:
            uq_method = "by propagating noise realisations through the model"
        parser.add_argument(
            "--get-initial-bounds", action='store_true',
            default=argparse.SUPPRESS,
            help=(
                f"Get pre-calibration bounds {uq_method}."
            )
        )
        parser.add_argument(
            "--n-noise-reals-per-img", type=int,
            default=argparse.SUPPRESS,
            help=(
                "Number of noise realizations per image. "
                f"Default = {_commons.N_NOISE_REALS_UQ}"
            )
        )
    if not zero_init_bounds:
        parser.add_argument(
            "-rho", "--hyperparam-precalib", type=float, nargs='+',
            default=argparse.SUPPRESS,
            help=(
                "Pre-calibration hyperparameter for CQR "
                "(multiplicative factor if `--mode-cqr` is set to 'addcqr', "
                "additive constant if `--mode-cqr` is set to 'multcqr'). "
                "Several value can be provided. Default = None"
            )
        )
        parser.add_argument(
            "--find-optimal-hyperparam-precalib", action='store_true',
            default=argparse.SUPPRESS
        )


def step_size_niter(parser, default_niter):

    parser.add_argument(
        "-tau", "--step-size", type=float, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            "Step size for the PnPMass algorithm. Several values can be provided. "
            "If not provided or set to 0, the step size will be computed as "
            f"Default = (1 - {_commons.EPS_SUP_STEP_SIZE:.1e}) * upper_bound, "
            "where upper_bound is estimated from the noise standard deviation "
            "and the mask, using the power iteration method."
        )
    )
    parser.add_argument(
        "-alph", "--multfact-step-size", type=float, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            "Multiplicative factor for the step size. "
            "Several values can be provided. "
            "Default = 1."
        )
    )
    parser.add_argument(
        "-i", "--niter", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of iterations for MCALens. "
            f"Default = {default_niter}"
        )
    )


def gaussian_extractor(parser, wiener=False, mcalens=False, verbose=False):

    additional_msg = (
        "Works with `--mode residual` or `--mode pnpmcalens`. "
    ) if verbose else ""
    parser.add_argument(
        "-ps", "--path-to-ps", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the power spectrum file. "
            f"{additional_msg}"
            f"Default = '{wl.PATH_TO_PS}'"
        )
    )
    if wiener:
        parser.add_argument(
            "--niter-wiener", type=int,
            default=argparse.SUPPRESS,
            help=(
                "Number of iterations for the Wiener filter. "
                f"{additional_msg}"
                f"Default = {_commons.NITER_WIENER}"
            )
        )
    if mcalens:
        additional_msg = (
            "Works with `--mode residual --which-gaussian-extractor mcalens` "
            "or `--mode pnpmcalens`. "
        ) if verbose else ""
        parser.add_argument(
            "-ig", "--niter-per-step-g", type=int,
            default=argparse.SUPPRESS,
            help=(
                "Number of iterations for one Gaussian step. "
                f"{additional_msg}"
                f"Default = {_commons.NITER_PER_STEP_G}"
            )
        )
        parser.add_argument(
            "-ing", "--niter-per-step-ng", type=int,
            default=argparse.SUPPRESS,
            help=(
                "Number of iterations for one non-Gaussian step. "
                f"{additional_msg}"
                f"Default = {_commons.NITER_PER_STEP_NG}"
            )
        )
        additional_msg = (
            "Works with `--mode residual --which-gaussian-extractor mcalens`. "
        ) if verbose else ""
        parser.add_argument(
            "-thresh", "--starlet-detection-threshold", type=float,
            default=argparse.SUPPRESS,
            help=(
                "Detection threshold for computing the support of active "
                "starlet coefficients. "
                f"{additional_msg}"
                f"Default = {int(_commons.STARLET_DETECTION_THRESHOLD)}-sigma"
            )
        )


def output(parser, prefix=None):

    parser.add_argument(
        "-o", "--output-prefix", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Prefix for output filename (without extension). "
            f"Default = '{prefix if prefix is not None else ""}'"
        )
    )
    parser.add_argument(
        "--save-tensors", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "If set, the tensors of the true convergence, "
            "the kappa map estimate, the variance, and the residuals "
            "will be saved in the output file. WARNING: this will increase "
            "the size of the output file significantly!"
        )
    )
    parser.add_argument(
        "--nimgs-save", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images to save. "
            f"Default = {_commons.NIMGS_SAVE}"
        )
    )


def seed_verbose(parser):

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


def _argument_exists(parser, flag):
    for action in parser._actions:
        if flag in action.option_strings:
            return True
    return False
