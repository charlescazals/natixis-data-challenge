"""
This modules contains function used in setup.py to link blob_credentials.py and load data.
"""
import os
from pathlib import Path

from azure.storage.blob import ContainerClient

from blob_credentials import facts_sas_token
from blob_credentials import facts_container

def link_credentials():
    """Function to create a symbolic link to blob_credentials.py.
    """
    os.system('ln -s ../credentials/blob_credentials.py')
    return 0


def load_data():
    """Function to load the input data from blob storage.
    """

    account_url = "https://hecdf.blob.core.windows.net"

    facts_blob_service = ContainerClient(account_url=account_url,
                                         container_name=facts_container,
                                         credential=facts_sas_token)

    print('````````````````````````````````````')
    print('        Begin loding data...')
    print('````````````````````````````````````')

    for blob in list(facts_blob_service.list_blobs()):
        file_name = blob.name
        print(file_name)
        download_stream = facts_blob_service.get_blob_client(file_name).download_blob()

        Path(f'./data/raw_in/{file_name}').parent.mkdir(parents=True, exist_ok=True)

        with open(f"./data/raw_in/{file_name}", "wb") as data:
            data.write(download_stream.readall())

    print('````````````````````````````````````')
    print('        Finished loading data!')
    print('````````````````````````````````````')

    return 0
