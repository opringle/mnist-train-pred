import argparse
import logging
import mxnet as mx
from mxnet import nd, gluon, autograd
import numpy as np
import multiprocessing
import os
from src import model


def evaluate_accuracy(data_iterator, net, ctx):
    """
    :param data_iterator: gluon data loader
    :param net: gluon hybrid sequential block
    :return: network accuracy on data
    """
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


def train(hyperparameters, channel_input_dirs, num_gpus, **kwargs):
    """
    train function formatted for use with amazon sagemaker
    :param hyperparameters: dict of network hyperparams
    :param channel_input_dirs: dict of paths to train and val data
    :param num_gpus: number of gpus to distribute training on
    :return: gluon neural network
    """
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    logging.info("Training context: {}".format(ctx))
    batch_size = hyperparameters.get('batch_size', 32)

    logging.info("Downloading the MNIST dataset, building train/test data loaders")
    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32) / 255, label)),
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count(),
        shuffle=True)
    test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32) / 255, label)),
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count(),
        shuffle=False)

    logging.info("Defining network architecture")
    net = model.CnnClassifier(dropout=0, num_label=10)
    logging.info("Network architecture: {}".format(net))

    if not hyperparameters.get('no_hybridize', False):
        logging.info("Hybridizing network to convert from imperitive to symbolic for increased training speed")
        net.hybridize()

    logging.info("Initializing network parameters with Xavier Algorithm")
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    optimizer = gluon.Trainer(params=net.collect_params(), optimizer='sgd',
                              optimizer_params={'learning_rate': hyperparameters.get('lr', 0.001),
                                                'momentum': hyperparameters.get('momentum', 0.9),
                                                'wd': hyperparameters.get('l2', 0.0)})

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    sm_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    epochs = hyperparameters.get('epochs', 10)
    logging.info("Training for {} epochs".format(epochs))
    for e in range(epochs):
        epoch_loss = 0
        weight_updates = 0
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                pred = net(data)
                loss = sm_loss(pred, label)
                loss.backward()
            optimizer.step(data.shape[0])
            epoch_loss += nd.mean(loss).asscalar()
            weight_updates += 1
        train_accuracy = evaluate_accuracy(train_data, net, ctx)
        val_accuracy = evaluate_accuracy(test_data, net, ctx)
        net.save_parameters(os.path.join('checkpoint', 'epoch' + str(e) + '.params'))
        logging.info("Epoch {}: Train Loss = {:.4} Train Accuracy = {:.4} Validation Accuracy = {:.4}".
                     format(e, epoch_loss / weight_updates, train_accuracy, val_accuracy))
    return net


if __name__ == "__main__":

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Train a CNN for digit classification")

    # Computation
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--gpus', type=int, default=0,
                        help='num of gpus to distribute  model training on. 0 for cpu')
    group.add_argument('--no-hybridize', action='store_true',
                        help='use symbolic network graph for increased computational eff')

    # Regularization
    group = parser.add_argument_group('Regularization arguments')
    group.add_argument('--dropout', type=float,
                       help='dropout probability for fully connected layers')
    group.add_argument('--l2', type=float,
                       help='weight regularization penalty')

    # Optimizer
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--epochs', type=int,
                       help='num of times to loop through training data')
    group.add_argument('--lr', type=float,
                       help='learning rate')
    group.add_argument('--momentum', type=float,
                       help='optimizer momentum')
    group.add_argument('--batch-size', type=int,
                       help='number of training examples per batch')

    args = parser.parse_args()
    hyp = {k: v for k, v in vars(args).items() if v is not None}
    train(hyperparameters=hyp, channel_input_dirs={'train': None, 'val': None}, num_gpus=args.gpus)
