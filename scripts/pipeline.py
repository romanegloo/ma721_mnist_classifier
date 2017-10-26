#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from mnist_x import DEFAULTS
from mnist_x import data, models

logger = logging.getLogger()


def set_defaults(args):
    # set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    logfile = logging.FileHandler(args.log_file, 'w')
    logfile.setFormatter(fmt)  # use the same format
    logger.addHandler(logfile)

    # change input dimensions, if dataset3 (3-digits) is used
    if args.dataset_name == 'dataset3':
        args.input_dim = 3 * 28 * 28
        args.output_dim = 1000
    elif args.dataset_name == 'dataset4':
        args.output_dim = 55

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')


# ------------------------------------------------------------------------------
# Train/Test Loop
# ------------------------------------------------------------------------------


def train(loader_train, loader_valid, model, optimizer, global_stats):
    """Run over one epoch of training model with the provided data loader"""
    model.train()

    correct = 0
    for idx, ex in enumerate(loader_train):
        images, labels = ex
        if args.cuda:
            x = Variable(images.cuda(async=True))
            y = Variable(labels.cuda(async=True))
        else:
            x, y = Variable(images), Variable(labels)
        y_pred = model(x)
        loss = F.nll_loss(y_pred, y, size_average=False)
        pred = y_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % 100 == 0:
            logger.info('iter = {}/{} | loss = {:.2f}'
                        ''.format(idx + 1, len(loader_train), loss.data[0]))
    train_accuracy = 100. * correct / len(loader_train.sampler)
    logger.info("== Epoch {} ==".format(global_stats['epoch']))
    global_stats['train_accuracy'].append(train_accuracy)
    logger.info("Training set || Accuracy: {}/{} ({:.2f}%)"
                ''.format(correct, len(loader_train.sampler), train_accuracy))

    # validate
    model.eval()
    total_loss = 0
    correct = 0
    len_v = len(loader_valid.sampler)
    for idx, ex in enumerate(loader_valid):
        images, labels = ex
        if args.cuda:
            x = Variable(images.cuda(async=True))
            y = Variable(labels.cuda(async=True))
        else:
            x, y = Variable(images), Variable(labels)
        y_pred = model(x)
        loss = F.nll_loss(y_pred, y, size_average=False)
        pred = y_pred.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).sum()
        total_loss += loss.data[0]
    loss = total_loss / len_v
    accuracy = 100. * correct / len_v
    global_stats['valid_losses'].append(loss)
    global_stats['valid_accuracy'].append(accuracy)
    logger.info("Validation set || Average_loss: {:.4f}, "
                "Accuracy: {}/{} ({:.2f}%)"
                ''.format(loss, correct, len_v, accuracy))


def test(loader_test, model, global_stats):
    model.eval()
    test_loss = 0
    correct = 0
    sample = None  # sample images for last verification
    for img, target in loader_test:
        if args.cuda:
            img = Variable(img.cuda(async=True), volatile=True)
            target = Variable(target.cuda(async=True))
        else:
            img, target = Variable(img, volatile=True), Variable(target)
        y_pred = model(img)
        test_loss += F.nll_loss(y_pred, target, size_average=False).data[0]
        pred = y_pred.data.max(1, keepdim=True)[1]
        sample = (img.data[0], target.data[0], pred[0])
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader_test.dataset)
    accuracy = 100. * correct / len(loader_test.dataset)
    global_stats['test_losses'].append(test_loss)
    global_stats['test_accuracy'].append(accuracy)

    logger.info("Test set || Average_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)"
                "".format(test_loss, correct, len(loader_test.dataset),
                          accuracy))

    if args.draw_image:
        logger.info("truth: {}, predicted: {}".format(sample[1], sample[2][0]))
        data.DataLoader.draw(sample[0])


def get_random_params():
    global args
    params = {}

    # minibatch_size
    args.batch_size = 2 ** np.random.randint(4, 11)
    params['batch_size'] = args.batch_size

    # number of hidden layers
    args.num_hidden_layers = 2 ** np.random.randint(6)
    params['num_hidden_layers'] = args.num_hidden_layers

    # number of hidden units
    args.num_hidden_units = 2 ** np.random.randint(3, 11)
    params['num_hidden_units'] = args.num_hidden_units

    # learning rate
    args.learning_rate = np.random.choice(np.logspace(-2, -6, 10))
    params['learning_rate'] = args.learning_rate

    # weight decay
    args.weight_decay = np.random.choice(np.linspace(0, 0.5, 5))
    params['weight_decay'] = args.weight_decay

    # initializer
    args.weight_init = np.random.choice(['none', 'uniform', 'xavier_normal'])
    params['weight_init'] = args.weight_init

    return params


# ------------------------------------------------------------------------------
# RUN!
# ------------------------------------------------------------------------------


def run(args, dataloaders):
    (loader_train, loader_valid, loader_test) = dataloaders

    # --------------------------------------------------------------------------
    # Model
    model = models.DynamicNet(args).cuda(args.gpu) if args.cuda \
        else models.DynamicNet(args)
    if args.parallel:
        model = torch.nn.DataParallel(model)
    model_summary, total_params = models.torch_summarize(model)
    if args.print_parameters:
         print(model_summary)

    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay)

    # --------------------------------------------------------------------------
    # Train/Test Loop
    logger.info('-' * 80)
    logger.info('Start training...')
    stats = {
        'epoch': 0,
        'best_ratio': 0,
        'best_accuracy': 0,
        'train_accuracy': [],
        'valid_losses': [],
        'valid_accuracy': [],
        'test_losses': [],
        'test_accuracy': [],
    }
    patience = 5
        # early stop training when ratio does not improve in this many times

    for epoch in range(1, args.num_epochs + 1):
        stats['epoch'] = epoch
        try:
            train(loader_train, loader_valid, model, optimizer, stats)
            test(loader_test, model, stats)
        except KeyboardInterrupt:
            logger.info(stats)
            raise

        # accuracy ratio to model capacity
        ratio = stats['test_accuracy'][-1] / (total_params * epoch)**.1
        if ratio > stats['best_ratio']:
            stats['best_ratio'] = ratio
            stats['best_accuracy'] = stats['test_accuracy'][-1]
            patience = 5
        else:
            patience -= 1
        logger.info('ratio to capacity: {} / {} = {:.4f}'
                    ''.format(stats['test_accuracy'][-1],
                              total_params * epoch, ratio))
        if patience == 0 and args.early_stop:
            break
        if np.isnan(stats['valid_losses'][-1]):
            break

    return stats


if __name__ == '__main__':
    # set up logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # parser
    parser = argparse.ArgumentParser(
        'MNIST Handwritten Digits Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Runtime options
    runtime = parser.add_argument_group('Runtime')
    runtime.add_argument('--num-epochs', type=int, default=50,
                        help='Number of full data iterations')
    runtime.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    runtime.add_argument('--data-workders', type=int, default=4,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--dataset-name', type=str, default='original',
                         choices=['original', 'dataset1', 'dataset2',
                                  'dataset3', 'dataset4'],
                         help='Name of modified datasets')
    runtime.add_argument('--num-random-models', type=int, default=1,
                         help='Number of random searches over hyper parameter '
                              'space')
    runtime.add_argument('--early-stop', action='store_true',
                         help='Stop training when no improvements seen in 3'
                         'times')
    runtime.add_argument('--no-cuda', action='store_true',
                         help='Use CPU only')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help="Specify GPU device id to use")
    runtime.add_argument('--parallel', type=bool, default=False,
                         help='Use DataParallel on all available GPUs')

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
    model.add_argument('--model-name', type=str, default=None,
                       help='Unique model identifier')
    model.add_argument('--input-dim', type=int, default=28*28,
                       help='input data dimension')
    model.add_argument('--output-dim', type=int, default=10,
                       help='output dimension (num. of classes)')
    model.add_argument('--num-hidden-units', type=int, default=64,
                       help='Dimension of hidden layer')
    model.add_argument('--num-hidden-layers', type=int, default=1,
                       help='Number of hidden layers')
    model.add_argument('--optimizer', type=str, default='sgd',
                       help='Optimizer: [sgd, adamax, adam]')
    model.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate for optimizer')
    model.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay, as L2 regularization by default')
    model.add_argument('--weight-init', type=str, default='none',
                       help='Add weight initialization scheme',
                       choices=['none', 'uniform', 'xavier_normal'])

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--print-parameters', action='store_true',
                         help='Print model parameters')
    runtime.add_argument('--draw-image', action='store_true',
                         help='draw images after test loop for verification')
    general.add_argument('--plot-losses', action='store_true',
                         help='plot train/test losses to epochs')
    general.add_argument('--log-file', action='store_true',
                         help='write logging on a file')
    args = parser.parse_args()
    set_defaults(args)

    # --------------------------------------------------------------------------
    # Loading Data
    logger.info('-' * 80)
    logger.info('Load data files')
    loader = data.DataLoader(args)
    (loader_train, loader_valid) = loader.load_training_torch()
    loader_test = loader.load_test_torch()

    best_model_ratio = 0
    best_model = None
    best_stats = None
    for m in range(args.num_random_models):
        logger.info('-' * 80)
        logger.info('Model #{}'.format(m+1))
        if m > 0:
            h_params = get_random_params()  # set random parameters
        else:  # run with the default (or given) settings first
            h_params = {
                'batch_size': args.batch_size,
                'num_hidden_layers': args.num_hidden_layers,
                'num_hidden_units': args.num_hidden_units,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'weight_init': args.weight_init
            }
        logger.info("{}".format(h_params))
        stats = run(args, (loader_train, loader_valid, loader_test))  # RUN ~!
        if stats['best_ratio'] >= best_model_ratio:
            best_model_ratio = stats['best_ratio']
            best_model = h_params
            best_stats = stats
            logger.info('Best Model Updated... (ratio: {:.2f})'
                        ''.format(best_model_ratio))

    # --------------------------------------------------------------------------
    # Display Results
    logger.info('-' * 80)
    logger.info('Best Model...')
    logger.info(best_model)
    logger.info(best_stats)

    if args.plot_losses:
        import matplotlib.pyplot as plt
        x = list(range(args.num_epochs))
        plt.plot(x, best_model['valid_losses'], 'g', label='train')
        plt.plot(x, best_model['test_losses'], 'r', label='test')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Test/Train Losses')
        plt.legend()
        plt.show()
