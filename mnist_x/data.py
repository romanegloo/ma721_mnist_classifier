# title: MA721 project 1 (MNIST classifier)
# filename: /scripts/prep_datasets/download_files.py
# author: Jiho Noh (jiho@cs.uky.edu)
"""from downloaded MNIST data files, build modified test/training datasets"""
import os
import struct
import numpy as np
import logging
from array import array
import torch
import torch.utils.data as utils

# set up logger
logger = logging.getLogger()


class DataLoader(object):
    def __init__(self, args):
        self.train_img_file = os.path.join(args.data_dir, args.train_img_file)
        self.train_lbl_file = os.path.join(args.data_dir, args.train_lbl_file)
        self.test_img_file = os.path.join(args.data_dir, args.test_img_file)
        self.test_lbl_file = os.path.join(args.data_dir, args.test_lbl_file)
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.data_workders
        self.shuffle = True

        # check dataset files exist
        files = [self.train_img_file, self.train_lbl_file,
                 self.test_img_file, self.test_lbl_file]
        for file in files:
            if not os.path.isfile(file):
                raise IOError("data file not exist {}".format(file))

    def load_training_np(self):
        images, labels = self._load('train')
        # self.train_images = self.process_images(images)
        # self.train_labels = self.process_labels(labels)
        return images, labels

    def load_training_torch(self):
        images, labels = self._load('train')
        dataset = utils.TensorDataset(torch.from_numpy(images).float(),
                                      torch.from_numpy(labels))
        return utils.DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers)

    def load_testing_np(self):
        images, labels = self._load('test')
        # self.test_images = self.process_images(images)
        # self.test_labels = self.process_labels(labels)
        return images, labels

    def load_test_torch(self):
        images, labels = self._load('test')
        dataset = utils.TensorDataset(torch.from_numpy(images).float(),
                                      torch.from_numpy(labels))
        # self.test_images = self.process_images(images)
        # self.test_labels = self.process_labels(labels)
        return utils.DataLoader(dataset,
                                batch_size=self.test_batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers)

    def process_images(self, images):
        pass

    def process_labels(self, labels):
        pass

    def _load(self, dataset_type='train'):
        if dataset_type == 'train':
            label_file = self.train_lbl_file
            image_file = self.train_img_file
        else:
            label_file = self.test_lbl_file
            image_file = self.test_img_file
        images, labels = [], []
        # refer to MNIST file format (http://yann.lecun.com/exdb/mnist/)
        # python struct (https://docs.python.org/3.5/library/struct.html)
        with open(label_file, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError("Magic number mismatch")
            labels = np.array(array('B', file.read()))

        with open(image_file, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError("Magic number mismatch")
            logging.info("loading images in [{} x {} x {}]"
                         "".format(size, rows, cols))
            images = np.fromfile(file, dtype=np.uint8).\
                reshape((size, rows * cols)).astype(np.int)
        return images, labels

    def draw(self, image):
        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()
