"""Download samples and eval data"""
import argparse
import os
import yaml
import wget
import gdown
import tarfile


def download_file(url, folder_path, file_name=None, extract=False, g_drive=False):
    """Download file from internet"""
    if g_drive:
        output_path = os.path.join(folder_path, file_name)
        gdown.download(url, output_path, quiet=False, fuzzy=True)
    else:
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
    if "dataset_url" in params:
        download_file(params["dataset_url"], output_path, extract=True)
    if "metadata_url" in params:
        download_file(params["metadata_url"], output_path)
    if "en_embeddings_url" in params:
        download_file(params["en_embeddings_url"], output_path, 
            file_name="dataperf_en_data.tar.gz", extract=True, g_drive=True)
    if "id_embeddings_url" in params:
        download_file(params["id_embeddings_url"], output_path, 
            file_name="dataperf_id_data.tar.gz", extract=True, g_drive=True)
    if "pt_embeddings_url" in params:
        download_file(params["pt_embeddings_url"], output_path, 
            file_name="dataperf_pt_data.tar.gz", extract=True, g_drive=True)

if __name__ == "__main__":
    main()
