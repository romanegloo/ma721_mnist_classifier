#!/usr/bin/env python3
# title: MA721 project 1 (MNIST classifier)
# filename: /scripts/prep_datasets/download_files.py
# author: Jiho Noh (jiho@cs.uky.edu)
"""downloads MNIST datasets"""

from tqdm import tqdm
import requests
import os
import logging
import subprocess

# set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# from Yan LeCun's website (http://yann.lecun.com/exdb/mnist/)
path_original_files = {
    'train-img': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train-lbl': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test-img': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test-lbl': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}
# path to data directory
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cur_dir, '..', 'data')


def _download_file(filetype, url, filename):
    logging.info("downloading {} from ({})".format(filetype, url))
    response = requests.get(url, stream=True)
    with open(os.path.join(data_dir, filename), "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)


def _decomp_gzip(filename):
    cmd = ['gzip', '-d', os.path.join(data_dir, filename)]
    logger.info('decompressing {}'.format(filename))
    subprocess.call(cmd)


if __name__ == '__main__':
    # check if files exist
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for filetype, url in path_original_files.items():
        filename = os.path.basename(url)
        if not os.path.isfile(os.path.join(data_dir, filename)):
            _download_file(filetype, url, filename)
            _decomp_gzip(filename)
