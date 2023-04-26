"""Download samples and eval data"""
import argparse
import os
import wget
import subprocess



def download_file(url, folder_path, lang_folder_path, extract=False):
    zip_path = wget.download(url, out=folder_path)
    extract_path = os.path.join(folder_path, lang_folder_path)
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    if extract:
        #unzip file using shell command to avoid character encoding issues with python zipfile lib
        subprocess.call(["unzip", zip_path, "-d", extract_path])


def main():

    # TODO: add URLs for each language dataset
    urls = {"en_embeddings_url": "http://dataperf.s3.amazonaws.com/speech-selection/dataperf_eng_data.zip",
                "id_embeddings_url": "http://dataperf.s3.amazonaws.com/speech-selection/dataperf_ind_data.zip",
                "pt_embeddings_url": "http://dataperf.s3.amazonaws.com/speech-selection/dataperf_por_data.zip"}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path where data will be stored",
    )
    args = parser.parse_args()
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    download_file(urls["en_embeddings_url"], output_path, "dataperf_en_data", extract=True)
    download_file(urls["id_embeddings_url"], output_path, "dataperf_id_data", extract=True)
    download_file(urls["pt_embeddings_url"], output_path, "dataperf_pt_data", extract=True)

if __name__ == "__main__":
    main()
