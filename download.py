import argparse
import sys
import os
from huggingface_hub import snapshot_download

__all_datasets__ = [
    "GTZAN",
    "EMO",
    "GS",
    "MTT",
    "HookTheory",
]

def download_dataset(dataset_name: str, save_root: str):
    """
    Download a single dataset and save its 'Data' folder directly under save_root/<dataset_name>/Data.
    """
    repo_id = f"m-a-p/{dataset_name}"
    target_dir = os.path.join(save_root, dataset_name)
    print(f"Starting download of dataset '{dataset_name}' into '{target_dir}'...")

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Download only the 'Data' subfolder, writing files directly into target_dir/Data
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )

    print(f"Dataset '{dataset_name}' Data folder has been saved to '{target_dir}/Data'.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache specified Hugging Face datasets (or 'all' for every supported one)."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help=f"Name of dataset to download (supported: {', '.join(__all_datasets__)}) or 'all' to download everything"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data",
        help="Root directory under which to save datasets (default: ./data)"
    )
    args = parser.parse_args()

    if args.dataset.lower() == "all":
        for ds in __all_datasets__:
            download_dataset(ds, args.save_dir)
    else:
        ds = args.dataset
        if ds not in __all_datasets__:
            print(
                f"Error: Dataset '{ds}' is not supported. Choose from: {', '.join(__all_datasets__)}",
                file=sys.stderr
            )
            sys.exit(1)
        download_dataset(ds, args.save_dir)


if __name__ == "__main__":
    main()
