#!/usr/bin/env python
## Based on https://github.com/chainer/chainer-chemistry/tree/master/examples/own_dataset

from __future__ import print_function

import chainer
import numpy
import os

from argparse import ArgumentParser
from chainer.datasets import split_dataset_random
from chainer import functions as F
from chainer.training import extensions
from chainerui.utils import save_args
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
from chainer.dataset import convert

from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.converters import converter_method_dict
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.links.scaler.standard_scaler import StandardScaler
from chainer_chemistry.models import Regressor
from chainer_chemistry.models.prediction import set_up_predictor, Classifier
from chainer_chemistry.utils import run_train, save_json
from chainer_chemistry.training.extensions import ROCAUCEvaluator  # NOQA
from chainer_chemistry.iterators.balanced_serial_iterator import BalancedSerialIterator  # NOQA
from chainer_chemistry.datasets import NumpyTupleDataset

def rmse(x0, x1):
    return F.sqrt(F.mean_squared_error(x0, x1))


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                    'relgat', 'gin', 'gnnfilm', 'relgcn_sparse', 'gin_sparse',
                    'nfp_gwm', 'ggnn_gwm', 'rsgcn_gwm', 'gin_gwm', 'megnet']    
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')
    parser.add_argument('--train', '-t', type=str,
                        default=None,
                        help='csv file containing the dataset')
    parser.add_argument('--val', '-v', type=str,
                        default='Np_test_1.csv',
                        help='csv file containing the dataset')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='relgcn')
    parser.add_argument('--method_suffix', '-ms', type=str, 
                        help='cwle or gwle', default='')
    parser.add_argument('--label', '-l', nargs='+',
                        default=['Np'],
                        help='target label for regression')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        help='label scaling method', default='standardize')
    parser.add_argument('--conv-layers', '-c', type=int, default=8,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=64,help='batch size')
    parser.add_argument('--device', '-g', type=str, default='0')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed model to')
    parser.add_argument('--epoch', '-e', type=int, default=800,
                        help='number of epochs')
    parser.add_argument('--unit-num', '-u', type=int, default=100,
                        help='number of units in one layer of the model')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--load_model', '-mf', type=str, default=None,
                        help='load saved model filename')
    parser.add_argument('--model-filename', type=str, default='model.pkl',
                        help='saved model filename')
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='path to a trainer snapshot')
    parser.add_argument('--balanced_iter', '-bi', action='store_true')
    parser.add_argument('--eval_roc', '-er', action='store_true')
    parser.add_argument('--classification', '-cl', action='store_true')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()
    args.out = os.path.join(args.out, args.method)
    save_args(args, args.out)

    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        raise ValueError('No target label was specified.')

    # Dataset preparation. Postprocessing is required for the regression task.
    def postprocess_label_float(label_list):
        return numpy.asarray(label_list, dtype=numpy.float32)
    def postprocess_label_int(label_list):
        return numpy.asarray(label_list, dtype=numpy.int64)

    # Apply a preprocessor to the dataset.
    if args.train:
    ## training data
        fn,ext = os.path.splitext(args.train)
        if ext==".npz":
            print('Loading training dataset...')
            train = NumpyTupleDataset.load(args.train)
        else:
            print('Preprocessing training dataset...')
            preprocessor = preprocess_method_dict[args.method]()
            if args.classification:
                parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label_int,labels=labels, smiles_col='SMILES')
            else:
                parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label_float,labels=labels, smiles_col='SMILES')
            train = parser.parse(args.train)['dataset']
            NumpyTupleDataset.save(os.path.join(args.out,os.path.split(fn)[1]), train)        
        # Scale the label values, if necessary.
        if args.scale == 'standardize':
            scaler = StandardScaler()
            scaler.fit(train.get_datasets()[-1])
        else:
            scaler = None

    ## test data
    fn,ext = os.path.splitext(args.val)
    if ext==".npz":
        print('Loading test dataset...')
        test = NumpyTupleDataset.load(args.val)
    else:
        print('Preprocessing test dataset...')
        preprocessor = preprocess_method_dict[args.method]()
        if args.classification:
            parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label_int,labels=labels, smiles_col='SMILES')
        else:
            parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label_float,labels=labels, smiles_col='SMILES')
        test = parser.parse(args.val)['dataset']
        NumpyTupleDataset.save(os.path.join(args.out,os.path.split(fn)[1]), test)


    # Set up the model.
    device = chainer.get_device(args.device)
    converter = converter_method_dict[args.method]
    metrics_fun = {'mae': F.mean_absolute_error, 'rmse': rmse}
    if args.classification:
        if args.load_model:
            model = Classifier.load_pickle(args.load_model, device=device)
            print("model file loaded: ",args.load_model)
        else:
            predictor = set_up_predictor(args.method, args.unit_num, args.conv_layers, class_num)
            model = Classifier(predictor,
                                    lossfun=F.sigmoid_cross_entropy,
                                    metrics_fun=F.binary_accuracy,
                                    device=device)
    else:
        if args.load_model:
            model = Regressor.load_pickle(args.load_model, device=device)
            print("model file loaded: ",args.load_model)
        else:
            predictor = set_up_predictor(
                args.method+args.method_suffix, args.unit_num,
                args.conv_layers, class_num, label_scaler=scaler)
            model = Regressor(predictor, lossfun=F.mean_squared_error,
                            metrics_fun=metrics_fun, device=device)

    if args.train:
        if args.balanced_iter:
            train = BalancedSerialIterator(train, args.batchsize, train.features[:, -1], ignore_labels=-1)
            train.show_label_stats()
            
        print('Training...')
        log_keys = ['main/mae','main/rmse','validation/main/mae','validation/main/rmse','validation/main/roc_auc']
        extensions_list = [extensions.PlotReport(log_keys, 'iteration', trigger=(100, 'iteration'), file_name='loss.png')]
        if args.eval_roc and args.classification:
            extensions_list.append(ROCAUCEvaluator(
                        test, model, eval_func=predictor,
                        device=device, converter=converter, name='validation',
                        pos_labels=1, ignore_labels=-1, raise_value_error=False))

        save_json(os.path.join(args.out, 'args.json'), vars(args))
        run_train(model, train, valid=test,
                batch_size=args.batchsize, epoch=args.epoch,
                out=args.out, extensions_list=extensions_list,
                device=device, converter=converter) #, resume_path=args.resume)

        # Save the model's parameters.
        model_path = os.path.join(args.out, args.model_filename)
        print('Saving the trained model to {}...'.format(model_path))
        if hasattr(model.predictor.graph_conv, 'reset_state'):
            model.predictor.graph_conv.reset_state()
        model.save_pickle(model_path, protocol=args.protocol)

    ## prediction
    it = SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    result = []
    for batch in it:
        in_arrays = convert._call_converter(converter, batch, device)
        with chainer.using_config('train', False), chainer.function.no_backprop_mode():
            if isinstance(in_arrays, tuple):
                res = model(*in_arrays)
            elif isinstance(in_arrays, dict):
                res = model(**in_arrays)
            else:
                res = model(in_arrays)
        result.extend(model.y.array.get())

    numpy.savetxt(os.path.join(args.out,"result.csv"), numpy.array(result))

    eval_result = Evaluator(it, model, converter=converter,device=device)()
    print('Evaluation result: ', eval_result)
#    save_json(os.path.join(args.in_dir, 'eval_result.json'), eval_result)
if __name__ == '__main__':
    main()
