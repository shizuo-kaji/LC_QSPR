#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @brief multi-layer perceptron for classification and regression
# @section Requirements:  python3,  chainer (pip install chainer)
# @version 0.01
# @date Oct. 2017
# @author Shizuo KAJI (shizuo.kaji@gmail.com)
# @licence MIT

from __future__ import print_function

import matplotlib.pyplot as plt
import argparse,os
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training,datasets,iterators
from chainer.training import extensions
import numpy as np
import pandas as pd
from chainer.dataset import dataset_mixin, convert, concat_examples

# activation function
activ = {
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'linear': F.identity,
    'relu': F.relu,
}

# Neural Network definition
class MLP(chainer.Chain):
    def __init__(self, args, std=1):
        super(MLP, self).__init__()
        self.activ=activ[args.activation]
        self.layers = args.layers
        self.out_ch = args.out_ch
        self.std = std
        self.dropout_ratio = args.dropout_ratio
#        self.add_link('norm{}'.format(0), L.BatchNormalization(args.in_ch))        
        self.add_link('layer{}'.format(0), L.Linear(None,args.unit))
        for i in range(1,self.layers):
            self.add_link('norm{}'.format(i), L.BatchNormalization(args.unit))        
            self.add_link('layer{}'.format(i), L.Linear(args.unit,args.unit))
        self.add_link('fin_layer', L.Linear(args.unit,args.out_ch))

    def __call__(self, x, t=0):
#        h = self['norm{}'.format(0)](x)
        h = self['layer{}'.format(0)](x)
        h = F.dropout(self.activ(h),ratio=self.dropout_ratio)
        for i in range(1,self.layers):
            h = self['norm{}'.format(i)](h)
            h = F.dropout(self.activ(self['layer{}'.format(i)](h)),ratio=self.dropout_ratio)
            #h = F.dropout(self.activ(self.bn(self.l2(h))))
        h = self['fin_layer'](h)
        if chainer.config.train:
            if self.out_ch > 1:    # classification
                loss = F.softmax_cross_entropy(h, t)
                chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            else:   #regression
                loss = F.mean_squared_error(t, h)
                MAE = self.std*F.mean_absolute_error(t,h)
                chainer.report({'loss': loss}, self)
                chainer.report({'MAE': MAE}, self)
            return loss
        return h


def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Multi-Perceptron classifier/regressor')
    parser.add_argument('dataset', help='Path to data file')
    parser.add_argument('--activation', '-a', choices=activ.keys(), default='sigmoid',
                        help='Activation function')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--dropout_ratio', '-dr', type=float, default=0,
                        help='dropout ratio')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--snapshot', '-s', type=int, default=-1,
                        help='snapshot interval')
    parser.add_argument('--label_index', '-l', type=int, default=5,
                        help='Column number of the target variable (5=Melting)')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--out_ch', '-oc', type=int, default=1,
                        help='num of output channels. set to 1 for regression')
    parser.add_argument('--optimizer', '-op', default='AdaDelta',
                        help='optimizer {MomentumSGD,AdaDelta,AdaGrad,Adam}')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--skip_columns', '-sc', type=int, default=29,
                        help='num of columns which are not used as explanatory variables')
    parser.add_argument('--layers', '-nl', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--unit', '-nu', type=int, default=100,
                        help='Number of units in the hidden layers')
    parser.add_argument('--test_every', '-t', type=int, default=5,
                        help='use one in every ? entries in the dataset for validation')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--weight_decay', '-w', type=float, default=0,
                        help='weight decay for regularization')
    args = parser.parse_args()
    args.regress = (args.out_ch == 1)

    # select numpy or cupy
    xp = chainer.cuda.cupy if args.gpu >= 0 else np
    label_type = np.int32 if not args.regress else np.float32

    # read csv file
    dat = pd.read_csv(args.dataset, header=0)

    ##
    print('Target: {}, GPU: {} Minibatch-size: {} # epoch: {}'.format(dat.keys()[args.label_index],args.gpu,args.batchsize,args.epoch))

#    csvdata = np.loadtxt(args.dataset, delimiter=",", skiprows=args.skip_rows)
    ind = np.ones(dat.shape[1], dtype=bool)  # indices for unused columns
    dat = dat.dropna(axis='columns')
    x = dat.iloc[:,args.skip_columns:].values
    args.in_ch = x.shape[1]
    t = (dat.iloc[:,args.label_index].values)[:,np.newaxis]
    print('target column:', args.label_index)
#    print('excluded columns: {}'.format(np.where(ind==False)[0].tolist()))
    print("data shape: ",x.shape, t.shape)
    x = np.array(x, dtype=np.float32)
    if args.regress:
        t = np.array(t, dtype=label_type)
    else:
        t = np.array(np.ndarray.flatten(t), dtype=label_type)

    # standardize
    t_mean = np.mean(t)
    t_std = np.std(t)
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = (x-x_mean)/x_std
    t = (t-t_mean)/t_std

    # Set up a neural network to train
    model = MLP(args,std=t_std)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimiser
    if args.optimizer == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD(lr=0.003, momentum=0.9)
    elif args.optimizer == 'AdaDelta':
        optimizer = chainer.optimizers.AdaDelta(rho=0.95, eps=1e-06)
    elif args.optimizer == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad(lr=0.001, eps=1e-08)
    elif args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    else:
        print("Wrong optimiser")
        exit(-1)
    optimizer.setup(model)
    if args.weight_decay>0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    print('layers: {}, units: {}, optimiser: {}, Weight decay: {}, dropout ratio: {}'.format(args.layers,args.unit,args.optimizer,args.weight_decay,args.dropout_ratio))


## train-validation data
# random spliting
    #train, test = datasets.split_dataset_random(datasets.TupleDataset(x, t), int(0.8*t.size))
# splitting by modulus of index
    train_idx = [i for i in range(t.size) if (i+1) % args.test_every != 0]
    var_idx = [i for i in range(t.size) if (i+1) % args.test_every == 0]
    n = len(train_idx)
    train_idx.extend(var_idx)
    train, test = datasets.split_dataset(datasets.TupleDataset(x, t), n, train_idx)

# dataset iterator
    train_iter = iterators.SerialIterator(train, args.batchsize, shuffle=True)
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    frequency = args.epoch if args.snapshot == -1 else max(1, args.snapshot)
    log_interval = 1, 'epoch'
    val_interval = frequency/10, 'epoch'

    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/MAE', 'validation/main/MAE',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']), trigger=log_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if not args.predict:
        trainer.run()
    else:
        test = datasets.TupleDataset(x, t)

    ## prediction
    print("predicting: {} entries...".format(len(test)))
    test_iter = iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    converter=concat_examples
    idx=0
    with open(os.path.join(args.outdir,'result.txt'),'w') as output:
        for batch in test_iter:
            x, t = converter(batch, device=args.gpu)
            with chainer.using_config('train', False):
                with chainer.function.no_backprop_mode():
                    if args.regress:
                        y = model(x).data
                        if args.gpu>-1:
                            y = xp.asnumpy(y)
                            t = xp.asnumpy(t)
                        y = y * t_std + t_mean
                        t = t * t_std + t_mean
                    else:
                        y = F.softmax(model(x)).data
                        if args.gpu>-1:
                            y = xp.asnumpy(y)
                            t = xp.asnumpy(t)
            for i in range(y.shape[0]):
                output.write(str(dat.iloc[var_idx[i],0]))
                if(len(t.shape)>1):
                    for j in range(t.shape[1]):
                        output.write(",{}".format(t[i,j]))
                        output.write(",{}".format(y[i,j]))
                else:
                    output.write(",{0:1.5f},{0:1.5f}".format(t[i],y[i]))
#                    output.write(",{0:1.5f}".format(np.argmax(y[i,:])))
#                    for yy in y[i]:
#                        output.write(",{0:1.5f}".format(yy))
                output.write("\n")
                idx += 1

        # rmse = F.mean_squared_error(pred,t)
        # result = np.vstack((tt,pred[:,0])).transpose()
        # # draw a graph
        # left = np.arange(len(test))
        # plt.plot(left, tt, color="royalblue")
        # plt.plot(left, pred[:,0], color="crimson", linestyle="dashed")
        # plt.title("RMSE: {}".format(np.sqrt(rmse.data)))
        # plt.show()

if __name__ == '__main__':
    main()
