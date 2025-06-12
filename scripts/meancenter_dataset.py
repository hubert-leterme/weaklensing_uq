import argparse
import wlmmuq.data.base_dataset as wlds

BATCH_SIZE = 256

def main(
        path_to_dataset, batch_size=BATCH_SIZE, verbose=False
):
    wlds.meancenter_dataset(
        path_to_dataset, batch_size=batch_size, verbose=verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_dataset", type=str,
        help="Path to the dataset (HDF5 file)"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size, to avoid memory overload. "
            f"Default = {BATCH_SIZE}"
        )
    )
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        default=argparse.SUPPRESS
    )

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
