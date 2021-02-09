#!/usr/bin/env python

from __future__ import print_function

import chainer
import numpy
import os

from argparse import ArgumentParser
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
from chainer.dataset import convert

from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.converters import converter_method_dict
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.models.prediction import set_up_predictor, Classifier
from chainer_chemistry.datasets import NumpyTupleDataset

# These imports are necessary for pickle to work.
from chainer_chemistry.models.prediction import GraphConvPredictor  # NOQA
from chainer_chemistry.utils import save_json

def rmse(x0, x1):
    return F.sqrt(F.mean_squared_error(x0, x1))

def parse_arguments():
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                    'relgat', 'gin', 'gnnfilm', 'relgcn_sparse', 'gin_sparse',
                    'nfp_gwm', 'ggnn_gwm', 'rsgcn_gwm', 'gin_gwm', 'megnet']
    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')
    parser.add_argument('--val', '-v', type=str,
                        default='dataset_test.csv',
                        help='csv file containing the dataset')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='relgcn')
    parser.add_argument('--label', '-l', nargs='+',
                        default=['Np'],
                        help='target label for regression')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--device', '-g', type=str, default='0')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed model to')
    parser.add_argument('--model-filename', '-mf', type=str, default='model.pkl',
                        help='saved model filename')
    parser.add_argument('--classification', '-cl', action='store_true')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()

    if args.label:
        labels = args.label
    else:
        raise ValueError('No target label was specified.')

    # Dataset preparation.
    def postprocess_label_float(label_list):
        return numpy.asarray(label_list, dtype=numpy.float32)
    def postprocess_label_int(label_list):
        return numpy.asarray(label_list, dtype=numpy.int64)

    fn,ext = os.path.splitext(args.val)
    if ext==".npz":
        print('Loading dataset...')
        test = NumpyTupleDataset.load(args.val)
    else:
        print('Preprocessing dataset...')
        preprocessor = preprocess_method_dict[args.method]()
        if args.classification:
            parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label_int,labels=labels, smiles_col='SMILES')
        else:
            parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label_float,labels=labels, smiles_col='SMILES')
        test = parser.parse(args.val)['dataset']

#    suc = dat['is_successful']

    print('Predicting...')
    # Set up the regressor.
    device = chainer.get_device(args.device)
    model_path = args.model_filename
    if args.classification:
        model =  Classifier.load_pickle(model_path, device=device)
    else:
        model = Regressor.load_pickle(model_path, device=device)

    converter = converter_method_dict[args.method]
    
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
