#!/usr/bin/env python3
"""
preprocessing.py

This script is designed to preprocess RNA secondary structure data. It takes raw data as input, processes 
it according to specified parameters, and outputs the preprocessed data. If no raw data is present, the 
script will download it from the specified URL.

NOTE: Generate and preprocess rna_sequences as List[str].

Usage:
    python run_preprocessing.py --input data/raw --output data/preprocessed_data.npy 

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
import argparse

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

def find_ct_files(input_dir: str) -> list:
    """
    Finds all .ct files in the given directory.
    
    :param input_dir: The directory where to look for CT files.
    :return: A list of paths to CT files.
    """
    # return [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".ct")]
    return [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if file.endswith(".ct")]

def extract_rna_sequence_from_ct(ct_file: str) -> tuple[str, str, int]:
    """
    Extracts the RNA sequence from a CT file. Evaluates if a pseudoknot 
    :param ct_file: Path to a CT file.
    :return: A tuple containing the filename (without extension) and the RNA sequence.
    """
    sequence = []
    with open(ct_file, 'r') as file:
        lines = file.readlines()
        # Skip the header line, process each nucleotide line
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            nucleotide = parts[1]
            sequence.append(nucleotide)
    filename = os.path.splitext(os.path.basename(ct_file))[0]
    # Return the filename without extension and the sequence
    return filename, ''.join(sequence)

def main():
    print("Preprocessing RNA secondary structure data...")

    # setup argument parsing
    parser = argparse.ArgumentParser(description="Preprocess RNA secondary structure data.")
    parser.add_argument("--url", type=str, help="URL to download RNA dataset from.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input raw data file.")
    parser.add_argument("--output", type=str, help="Path to the output preprocessed data file (in .npy format).")
    args = parser.parse_args()

    # download data if it does not exist
    if not os.listdir(args.input):
        download_data(args.url)
    else:
        print("Some data already exists. Skipping download.")
    
    ct_files = find_ct_files(args.input)
    print(len(ct_files), "CT files found.") # TODO: convert to logger


    print("Preprocessing complete.")
    
if __name__ == "__main__":
    main()


    