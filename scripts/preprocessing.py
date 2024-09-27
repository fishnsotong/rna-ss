#!/usr/bin/env python3
"""
preprocessing.py

This script is designed to preprocess RNA secondary structure data. It takes raw data as input, processes 
it according to specified parameters, and outputs the preprocessed data. If no raw data is present, the 
script will download it from the specified URL.

Usage:
    python run_preprocessing.py --input data/raw_data.npy --output data/preprocessed_data.npy --batch-size 64 --shuffle

Arguments:
    --input: Path to the input raw data file (in .npy format).
    --output: Path to the output preprocessed data file (in .npy format).
    --batch-size: Size of the batches for processing.
    --shuffle: Whether to shuffle the data before processing.

Author: Wayne Yeo
"""
import os
import requests
import tarfile

def download_data(url, output_dir="data/raw"):
    """
    Download raw data from a given URL and save it to the specified directory.

    :param url: URL to download the data from
    :param output_dir: Directory to save the downloaded data
    """
    # download the file, define file name based on URL
    print(f"Downloading data from {url} to {output_dir}...")
    r = requests.get(url, stream=True)
    tar_file_name = os.path.join(os.getcwd(), url.split("/")[-1])

    with open(tar_file_name, 'wb') as f:
        f.write(r.content)

    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # extract
    with tarfile.open(tar_file_name, 'r:gz') as tar:
        tar.extractall(path=output_dir)
    
    # remove the downloaded tar file after extraction
    os.remove(tar_file_name)

    print(f"Downloaded and extracted {tar_file_name} to {output_dir}")
    pass

def main():
    # Your main preprocessing logic here
    print("Preprocessing RNA secondary structure data...")
    download_data("https://rna.urmc.rochester.edu/pub/archiveII.tar.gz")

if __name__ == "__main__":
    main()