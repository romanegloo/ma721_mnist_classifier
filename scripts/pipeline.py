#!/usr/bin/env python3

import argparse
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import logging
import subprocess
import os

from mnist_x import data, models
from mnist_x import DEFAULTS

logger = logging.getLogger()


def set_defaults(args):
    # set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')


# ------------------------------------------------------------------------------
# Train/Test Loop
# ------------------------------------------------------------------------------


def train(data_loader, model, optimizer, global_stats):
    """Run over one epoch of training model with the provided data loader"""
    model.train()
    total_loss = 0
    for idx, ex in enumerate(data_loader):
        images, labels = ex
        x, y = Variable(images), Variable(labels)
        y_pred = model(x)
        loss = F.nll_loss(y_pred, y)
        total_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % 500 == 0:
            avg_loss = 0 if idx == 0 else total_loss / idx
            logger.info('train: Epoch = {} | iter = {}/{} | loss = {:.2f}'
                        ''.format(global_stats['epoch'], idx + 1,
                                  len(data_loader), avg_loss))


def test(data_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        y_pred = model(data)
        test_loss += F.nll_loss(y_pred, target, size_average=False).data[0]
        pred = y_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(data_loader.dataset)
    logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)"
                "".format(test_loss, correct, len(data_loader.dataset),
                          100. * correct / len(data_loader.dataset)))

# ------------------------------------------------------------------------------
# RUN!
# ------------------------------------------------------------------------------


def run(args):
    # --------------------------------------------------------------------------
    # Loading Data
    logger.info('-' * 100)
    logger.info('Load data files')
    loader = data.DataLoader(args)
    loader_train = loader.load_training_torch()
    loader_test = loader.load_test_torch()

    # --------------------------------------------------------------------------
    # Model
    model = models.DynamicNet(args)
    model_summary, total_params = models.torch_summarize(model)
    if args.print_parameters:
         print(model_summary)
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters())

    # --------------------------------------------------------------------------
    # Train/Test Loop
    logger.info('-' * 100)
    logger.info('Start training...')
    stats = {'epoch': 0, 'best_valid': 0}

    for epoch in range(0, args.num_epochs):
        stats['epoch'] = epoch
        train(loader_train, model, optimizer, stats)
        test(loader_test, model)

    # todo. at some point, I need to print out the performance results with
    # the accuracies (test/train), and the ratio to the (params * epochs)^(1/10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MNIST Handwritten Digits Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Runtime options (later add options for gpu use)
    runtime = parser.add_argument_group('Runtime')
    runtime.add_argument('--num-epochs', type=int, default=30,
                        help='Number of full data iterations')
    runtime.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                        help='Batch size for testing/validating')
    runtime.add_argument('--data-workders', type=int, default=4,
                         help='Number of subprocesses for data loading')

    # Files
    files = parser.add_argument_group('Filesystem')
    parser.add_argument('--model-dir', type=str, default=DEFAULTS['MODEL_DIR'],
                        help='Directory for saved models')
    parser.add_argument('--data-dir', type=str, default=DEFAULTS['DATA_DIR'],
                        help='directory of datasets')
    parser.add_argument('--train-img-file', type=str,
                        default='train-images-idx3-ubyte',
                        help='Decompressed MNIST train (images) file')
    parser.add_argument('--train-lbl-file', type=str,
                        default='train-labels-idx1-ubyte',
                        help='Decompressed MNIST train (labels) file')
    parser.add_argument('--test-img-file', type=str,
                        default='t10k-images-idx3-ubyte',
                        help='Decompressed MNIST test (images) file')
    parser.add_argument('--test-lbl-file', type=str,
                        default='t10k-labels-idx1-ubyte',
                        help='Decompressed MNIST test (labels) file')

    # Model architecture
    model = parser.add_argument_group('Model')
    model.add_argument('--model-type', type=str, default='2hdd',
                       help='Model architecture type')
    model.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier')
    model.add_argument('--input-dim', type=int, default=28*28,
                       help='input data dimension')
    model.add_argument('--output-dim', type=int, default=10,
                       help='output dimension (num. of classes)')
    model.add_argument('--hidden-dim', type=int, default=128,
                       help='Dimension of hidden layer')
    model.add_argument('--num-hidden-layers', type=int, default=1,
                       help='Number of hidden layers')
    model.add_argument('--optimizer', type=str, default='sgd',
                       help='Optimizer: [sgd, adamax]')
    model.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate for SGD')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--print-parameters', action='store_true',
                         help='Print model parameters')
    args = parser.parse_args()
    set_defaults(args)

    # set up logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # RUN !!
    run(args)

#     # test: loading, drawing a sample
#     # --------------------------------------------------------------------------
#     # loader = DataLoader()
#     # images, labels = loader.load_testing_np()
#     # loader.draw(images[0].reshape((28, 28)))
#     # print(labels[:10])
