"""
Module to load Romanian text data from local files.
"""
import os
from typing import List


def load_romanian_text(local_file: str) -> List[str]:
    """
    Load Romanian text from a local file.
    
    Args:
        local_file: Path to local text file
        
    Returns:
        List of Romanian text strings
    """
    return load_from_local_file(local_file)


def load_from_local_file(file_path: str) -> List[str]:
    """
    Load Romanian text from a local text file.
    
    Args:
        file_path: Path to the local text file
        
    Returns:
        List of Romanian text strings (one per line, non-empty lines only)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")
    
    print(f"Loading Romanian text from local file: {file_path}")
    texts = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:  # Skip empty lines
                texts.append(text)
    
    print(f"Loaded {len(texts)} Romanian text lines from file")
    return texts

