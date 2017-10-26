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
from torch.utils.data.sampler import SubsetRandomSampler

# set up logger
logger = logging.getLogger()


class DataLoader(object):
    def __init__(self, args):
        """
        Read dataset files and prepare DataLoader for neural network run.
        Performs preprocessing upon the dataset_name
        :param args: user provided + default arguemtns
        """
        self.train_img_file = os.path.join(args.data_dir, args.train_img_file)
        self.train_lbl_file = os.path.join(args.data_dir, args.train_lbl_file)
        self.test_img_file = os.path.join(args.data_dir, args.test_img_file)
        self.test_lbl_file = os.path.join(args.data_dir, args.test_lbl_file)
        self.batch_size = args.batch_size
        self.num_workers = args.data_workders
        self.shuffle = True
        self.dataset_name = args.dataset_name
        self.pin_memory = False  #args.cuda

        # check dataset files exist
        files = [self.train_img_file, self.train_lbl_file,
                 self.test_img_file, self.test_lbl_file]
        for file in files:
            if not os.path.isfile(file):
                msg = "Data file not found. Please check the path " +\
                      "or download files using scripts/download_files.py "
                raise IOError(msg)

    def load_training_torch(self):
        images, labels = self._load('train')
        images = self.process_images(images, 'train')
        labels = self.process_labels(labels, 'train')
        dataset = utils.TensorDataset(torch.from_numpy(images).float(),
                                      torch.from_numpy(labels))

        valid_size = 0.1
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        # shuffle
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        logger.info("dataLoader for train is prepared")
        loader_train = utils.DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        sampler=train_sampler,
                                        pin_memory=self.pin_memory)
        loader_valid = utils.DataLoader(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        sampler=valid_sampler,
                                        pin_memory=self.pin_memory)
        return (loader_train, loader_valid)

    def load_test_torch(self):
        if self.dataset_name == 'dataset1':
            images, labels = self._load('train')
        else:
            images, labels = self._load('test')
        images = self.process_images(images, 'test')
        labels = self.process_labels(labels, 'test')
        dataset = utils.TensorDataset(torch.from_numpy(images).float(),
                                      torch.from_numpy(labels))
        logger.info("dataLoader for test is prepared")
        return utils.DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory)

    def process_images(self, images, mode='train'):
        """preprocess input datasets, which are still numpy ndarrays"""
        if self.dataset_name == 'dataset1':
            return images[:5000]
        elif self.dataset_name == 'dataset2':
            return np.add(images, 745)
        elif self.dataset_name == 'dataset3':
            # concatenate three images into three-digit image
            if mode == 'train':
                return np.concatenate((images[:40000], images[10000:50000],
                                      images[20000:60000]), axis=1)
            elif mode == 'test':
                return np.concatenate((images[:8000], images[1000:9000],
                                       images[2000:10000]), axis=1)
        elif self.dataset_name == 'dataset4':
            # merge two images into one
            if mode == 'train':
                return images[:50000] + images[-50000:]
            elif mode == 'test':
                return images[:9000] + images[-9000:]
        else:
            return images

    def process_labels(self, labels, mode='train'):
        if self.dataset_name == 'dataset1':
            return labels[:5000]
        elif self.dataset_name == 'dataset2':
            return labels
        elif self.dataset_name == 'dataset3':
            if mode == 'train':
                return 100 * labels[:40000] + 10 * labels[10000:50000] +\
                    labels[20000:60000]
            elif mode == 'test':
                return 100 * labels[:8000] + 10 * labels[1000:9000] + \
                       labels[2000:10000]
        elif self.dataset_name == 'dataset4':
            """choosing two numbers without an order, label them 0..54
            let (m, n) be the numbers where m >= n,
                label l = m ( m + 1) / 2 + n
            """
            if mode == 'train':
                m = np.maximum(labels[:50000], labels[-50000:])
                n = np.minimum(labels[:50000], labels[-50000:])
                l = np.add(np.multiply(m, m + 1) / 2, n)
            elif mode == 'test':
                m = np.maximum(labels[:9000], labels[-9000:])
                n = np.minimum(labels[:9000], labels[-9000:])
                l = np.add(np.multiply(m, m + 1) / 2, n)
            return l.astype(np.int)
        else:
            return labels

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
            labels = np.array(array('B', file.read())).astype(np.int)

        with open(image_file, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError("Magic number mismatch")
            logger.info("loading images in [{} x {} x {}]"
                         "".format(size, rows, cols))
            images = np.fromfile(file, dtype=np.uint8).\
                reshape((size, rows * cols)).astype(np.int)
        return images, labels

    @classmethod
    def draw(cls, img):
        # draw given image data, assuming 1-d array (either 28*28 or 28*28*3)
        from matplotlib import pyplot as plt
        import matplotlib as mpl

        print(type(img))
        if 'numpy' in dir(img):
            img = img.numpy()
        img_dim = (28, 28)
        img_len = np.prod(img_dim)
        if len(img) == img_len:
            plt.imshow(img.reshape(img_dim), cmap=mpl.cm.Greys)
        elif len(img) == img_len * 3:
            img3 = np.concatenate((
                img[:img_len].reshape(img_dim),
                img[img_len:2*img_len].reshape(img_dim),
                img[2*img_len:3*img_len].reshape(img_dim)), axis=1)
            plt.imshow(img3, cmap=mpl.cm.Greys)
        else:
            logger.error("draw: unknow image size {}".format(img.size()))

        plt.show()
