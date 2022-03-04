"""Download samples and eval data"""
import argparse
import os
import yaml
import gdown


def download_file(url, folder_path, file_name):
    """Download file from Google Drive"""
    output_path = os.path.join(folder_path, file_name)
    gdown.download(url, output_path, quiet=False, fuzzy=True)


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
        "--output_path",
        type=str,
        required=True,
        help="Path where data will be stored",
    )
    args = parser.parse_args()

    with open(args.parameters_file, "r") as f:
        params = yaml.full_load(f)

    output_path = args.output_path
    samples_url = params["samples_url"]
    eval_url = params["eval_url"]

    download_file(samples_url, output_path, "samples.pb")
    download_file(eval_url, output_path, "eval.pb")


if __name__ == "__main__":
    main()
