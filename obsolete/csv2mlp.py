#!/usr/bin/env python
# based on csv2libsvm.py

import sys
import csv
import operator
from collections import defaultdict

import argparse
import random
import sys

def construct_line(label, line, labels_dict):
    new_line = []
    if label in labels_dict:
        new_line.append(labels_dict.get(label))
    else:
        label_id = str(len(labels_dict))
        labels_dict[label] = label_id
        new_line.append(label_id)

    for item in line:
        new_line.append(item)
    new_line = ",".join(new_line)
    return new_line

def main():
    parser = argparse.ArgumentParser(description='convert a csv to one which is compatible with MLP.py')
    parser.add_argument('filename', help='Path to csv file')
    parser.add_argument('--label_index', '-l', type=int, default=0,
                        help='column number of the target variable')
    parser.add_argument('--skip_headers', '-s', action='store_true')
    parser.add_argument('--skip_column', '-sc', type=int, nargs="*", default=[],
                        help='set of indices of columns to be skipped in the data')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    input_file = args.filename

    i = open(input_file, 'rU')

    reader = csv.reader(i)

    if args.skip_headers:
        next(reader)

    args.skip_column.append(args.label_index)
    sc=list(set(args.skip_column))
    sc.reverse()

    labels_dict = {}
    for line in reader:
        if len(line) <= args.label_index:
            continue
        label = line[args.label_index]
        for i in sc:
            line.pop(i)
        new_line = construct_line(label, line, labels_dict)
        print(new_line)

if __name__ == '__main__':
    main()

