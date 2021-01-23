"""
High level function to link credentials and load the input data (see utils_setup.py)
"""
import os
from utils_setup import link_credentials, load_data


def load():
    """Setup function to load the input data.

    Args:
        None

    Returns:
        None

    """
    if 'blob_credentials.py' not in os.listdir():
        link_credentials()

    load_data()


if __name__ == '__main__':
    load()
