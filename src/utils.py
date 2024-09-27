import os
import sys
import logging

"""
utils.py

This module contains utility functions for the RNA secondary structure project.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_file(file_path):
    """
    Reads the content of a file and returns it.
    
    :param file_path: Path to the file
    :return: Content of the file
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    return content

def write_file(file_path, content):
    """
    Writes content to a file.
    
    :param file_path: Path to the file
    :param content: Content to write
    """
    with open(file_path, 'w') as file:
        file.write(content)
    logger.info(f"Content written to {file_path}")

def validate_sequence(sequence):
    """
    Validates if a given sequence is a valid RNA sequence.
    
    :param sequence: RNA sequence
    :return: True if valid, False otherwise
    """
    valid_nucleotides = {'A', 'U', 'C', 'G'}
    is_valid = all(nucleotide in valid_nucleotides for nucleotide in sequence)
    
    if not is_valid:
        logger.warning(f"Invalid RNA sequence: {sequence}")
    
    return is_valid

# Add more utility functions as needed