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

def ct_to_dbn(ct_filename: str) -> str:
    """
    Converts a CT (Connectivity Table) file into a dot-bracket notation (DBN) string representing RNA 
    secondary structure.

    This function parses a CT file to extract the RNA sequence and base-pairing information, 
    then generates a corresponding dot-bracket notation (DBN) string. The DBN string uses 
    various types of brackets to indicate base pairs, while unpaired nucleotides are denoted by dots (`.`).

    Parameters:
    ----------
    ct_filename : str
        Path to the CT file that contains RNA sequence and base-pairing information.
        The CT file format contains nucleotide information and the indices of base-pairing partners.

    Returns:
    -------
    str
        A string in dot-bracket notation representing the RNA secondary structure.
        The string consists of dots (unpaired nucleotides) and various brackets (paired nucleotides).
        Different types of brackets (e.g., '()', '<>', '[]', '{}') represent nested and pseudoknotted structures.

    Example:
    --------
    >>> ct_to_dbn('example.ct')
    '..((..<<..>>..))..'

    Notes:
    ------
    - The CT file typically contains columns representing the nucleotide index, the nucleotide itself, 
      and the index of its paired nucleotide (or `0` if unpaired).
    - Pseudoknots and nested structures are represented by different levels of brackets ('()', '<>', '[]', '{}').
    - The function handles base-pairing information, deduplicates pairs, and assigns brackets at different 
      nesting levels.
    - If a pseudoknot or interleaving base pair is detected, the function advances to the next available 
      bracket level.
    - The number of bracket levels is limited by the predefined list (`levels`), which currently supports 
      up to four levels.

    """
    name = None
    sequence = []
    raw_pairlist = []

    with open(ct_filename, 'r') as f:
        for i, line in enumerate(f):
            if not i:
                name = line.split()[1]              # name of the ncRNA
            else:
                sequence.append(line.split()[1])    # save primary sequence to string

                n = int(line.split()[0])            # nucleotide index
                k = int(line.split()[4])            # base-pairing partner index
                
                # only considered paired bases (k != 0)
                if k > 0:
                    raw_pairlist.append((n, k))

    # deduplicate pairlist
    pairlist = []
    seen = set()

    for pair in raw_pairlist:
        sorted_pair = tuple(sorted(pair))
        if sorted_pair not in seen:
            seen.add(sorted_pair)
            pairlist.append(sorted_pair)

    dots = ['.'] * len(sequence)

    levels = ['()', '<>', '[]', '{}']
    current_level = 0

    for i, (start, end) in enumerate(pairlist):
        for _, (prev_start, prev_end) in enumerate(pairlist[:i]):
            # Is there any interleaving with previous pairs?
            if prev_start < start < prev_end < end:

                # Control advancement of the current_level but ensure it does not exceed the highest 
                # allowable level (determined by the length of the levels list).
                current_level = min(current_level + 1, len(levels) - 1)
                break
            else:
                current_level = 0

            # Use the current level's brackets for this base pair
        open_bracket, close_bracket = levels[current_level]
        if dots[start - 1] == "." and dots[end - 1] == ".":
            dots[start - 1] = open_bracket
            dots[end - 1] = close_bracket

    return "".join(dots)

def pseudoknot_checker(dbn: str) -> int:
    """
    Checks for the presence of pseudoknots in a dot-bracket notation (DBN) string.

    This function scans a dot-bracket notation (DBN) string to detect the presence of pseudoknots.
    A pseudoknot is typically indicated by the presence of angle brackets (`<` or `>`), which represent
    non-canonical base pairing interactions that are not nested within the usual dot-bracket structure.

    Parameters:
    ----------
    dbn : str
        A string representing RNA secondary structure in dot-bracket notation.
        Valid characters include '.', '(', ')', '[', ']', '<', '>', etc.

    Returns:
    -------
    int
        Returns 1 if pseudoknots (angle brackets '<' or '>') are detected in the DBN string, otherwise returns 0.

    Example:
    --------
    >>> dbn = "..((..<<..>>..)).."
    >>> pseudoknot_checker(dbn)
    1
    
    >>> dbn = "..((..)).."
    >>> pseudoknot_checker(dbn)
    0
    """
    pseudoknot_state = 0
    for i in dbn:
        # if "<" in dbn: 
        # even if there's a for loop in the background it's still more efficient
        # TODO: return "<" in dbn 
        if i in "<>":
            pseudoknot_state = 1
            break

    return pseudoknot_state

def add_labels_to_data(i):
    names, sequences = i
    labels = []
    for name, _ in zip(names, sequences):
        labels.append(pseudoknot_checker(ct_to_dbn("./gis_data/archiveII/" + name + ".ct")))
    return (names, sequences, labels)

def fasta_parse(fasta_file: str, comment='#'):
    """
    Parses a FASTA file and extracts sequence names and their corresponding sequences.

    The function reads a given FASTA file, where each sequence is identified by a line starting with 
    the '>' symbol, followed by the sequence name. The subsequent lines contain the sequence data, 
    which is concatenated into a single string for each sequence. The function also ignores any comment 
    lines that start with the specified comment character.

    Parameters:

    fasta_file : str
        The path to the FASTA file to be parsed.
    comment : str, optional
        A character indicating comment lines that should be ignored. Default is '#'.
    
    Returns:
    -------
    tuple:
        A tuple containing two lists:
        - names : List[str]
            A list of sequence names extracted from lines starting with '>'.
        - sequences : List[str]
            A list of corresponding sequences, with each sequence represented as a string.
    
    Example:
    --------
    >>> names, sequences = fasta_parse("example.fasta")
    >>> print(names)
    ['sequence1', 'sequence2']
    >>> print(sequences)
    ['ATCGATCG', 'GGCTAAGT']

    Notes:
    ------
    - Sequence names are taken from lines starting with '>', with the '>' character removed.
    - Sequences are converted to uppercase.
    - The function assumes that the sequences are stored in a standard FASTA format.

    """
    names = []
    sequences = []
    name = None
    sequence = []
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(comment):
                continue
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    names.append(name)
                    sequences.append(''.join(sequence))
                name = line[1:]
                sequence = []
            else:
                sequence.append(line.upper())
        if name is not None:
            names.append(name)
            sequences.append(''.join(sequence))

    return names, sequences

def dedup_sequences(data_tuple):
    """
    Removes duplicate sequences from a tuple containing sequence names and sequences.

    This function takes a tuple consisting of two lists: one for sequence names and 
    one for the corresponding sequences. It removes duplicate sequences and returns 
    a new tuple containing only the unique sequences and their corresponding names.

    Parameters:
    ----------
    data_tuple : tuple
        A tuple containing two lists:
        - names (list of str): A list of sequence names.
        - sequences (list of str): A list of nucleotide or protein sequences.

    Returns:
    -------
    tuple
        A tuple containing two lists:
        - result_names (list of str): A list of names corresponding to the unique sequences.
        - result_sequences (list of str): A list of unique sequences.

    Example:
    --------
    >>> names = ["seq1", "seq2", "seq3"]
    >>> sequences = ["AGCT", "CGTA", "AGCT"]
    >>> data_tuple = (names, sequences)
    >>> dedup_sequences(data_tuple)
    (["seq1", "seq2"], ["AGCT", "CGTA"])

    This function can also be modified to produce a tuple of duplicate values.
    """
    names, sequences = data_tuple
    unique_sequences = {}
    result_names = []
    result_sequences = []
    counter = 0
    
    for name, seq in zip(names, sequences):
        if seq not in unique_sequences:
            unique_sequences[seq] = name
            result_names.append(name)
            result_sequences.append(seq)
        else:
            print(f"Duplicate sequence found: {name} and {unique_sequences[seq]}")
            counter += 1

    print(f"\nTotal duplicate sequences found: {counter}")
    return (result_names, result_sequences)

def fasta_write(data_tuple, output_fasta):
    """
    Writes a tuple of names and sequences to a FASTA file.
    
    Parameters:
    ----------
    data_tuple : tuple
        A tuple where the first element is a list of names and the second element is a list of sequences.
    output_fasta : str
        The name of the output FASTA file.
    """
    names, sequences = data_tuple
    with open(output_fasta, 'w') as fasta_file:
        for name, sequence in zip(names, sequences):
            fasta_file.write(f">{name}\n{sequence}\n")
    print(f"Data written to {output_fasta}")