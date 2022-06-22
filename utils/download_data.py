"""Download samples and eval data"""
import argparse
import os
import yaml
import wget
import tarfile


def download_file(url, folder_path, extract=False):
    """Download file from internet"""
    output_path = wget.download(url, out=folder_path)
    if extract:
        tar = tarfile.open(output_path, "r:gz")
        tar.extractall(folder_path)
        tar.close()


def main():
    """Main function that perform the download"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameters_file",
        type=str,
        required=True,
        help="File containing parameters for the download",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path where data will be stored",
    )
    args = parser.parse_args()

    with open(args.parameters_file, "r") as f:
        params = yaml.full_load(f)

    output_path = args.output_path
    dataset_url = params["dataset_url"]
    metadata_url = params["metadata_url"]
    embeddings_url = params["embeddings_url"]

    download_file(metadata_url, output_path)
    download_file(dataset_url, output_path, extract=True)
    download_file(embeddings_url, output_path, extract=True)


if __name__ == "__main__":
    main()
