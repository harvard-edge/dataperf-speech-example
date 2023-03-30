"""Download samples and eval data"""
import argparse
import os
import yaml
import wget
import tarfile


def download_file(url, folder_path, extract=False):
    output_path = wget.download(url, out=folder_path)
    if extract:
        tar = tarfile.open(output_path, "r:gz")
        tar.extractall(folder_path)
        tar.close()


def main():

    # TODO: add URLs for each language dataset
    urls = dict()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path where data will be stored",
    )
    args = parser.parse_args()
    output_path = args.output_path
    download_file(urls["en_embeddings_url"], output_path, extract=True)
    download_file(urls["id_embeddings_url"], output_path, extract=True)
    download_file(urls["pt_embeddings_url"], output_path, extract=True)

if __name__ == "__main__":
    main()
