import requests
import zipfile
import os
import torch
import numpy as np

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

######################################################
def download_required():
    # download file from google
    file_id = '12HDIutE07f-TlHAXxAsHT-cv-fVQ_WZr'
    destination = 'hw3-train-validation.zip '
    download_file_from_google_drive(file_id, destination)

    # unzip data files
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall('dataset')
    zip_ref.close()

    # remove zip file
    os.remove(destination)

    # download file from google
    file_id = '1IKY0_LAg_-IzbmfCtUvv2ShWihAZaYos'
    destination = 'mean_iou_evaluate.py'
    download_file_from_google_drive(file_id, destination)
    
######################################################
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
    
    