#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Shizuo KAJI (shizuo.kaji@gmail.com)
# @licence MIT

from __future__ import print_function

import matplotlib.pyplot as plt
import argparse,os
import numpy as np
import pandas as pd

def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('datasets', nargs="*", help='Path to data file')
    parser.add_argument('--output', '-o', default="merged.csv", help='csv file name')

    args = parser.parse_args()

    # read csv file
    for i,fn in enumerate(args.datasets):
        print(fn)
        if(i==0):
            df = pd.read_csv(fn, header=0, index_col="ID")
        else:
            df2 = pd.read_csv(fn, header=0, index_col="ID")
        #    df2.sort_values(by=["ID"], ascending=True, inplace=True)
        #    df2.reset_index(inplace=True)
            df = df.merge(df2, on="ID", how="outer", suffixes=('', '_2'), copy=False, sort=True)
            df['SMILES'] = df['SMILES'].combine_first(df['SMILES_2'])
            df['Phases'] = df['Phases'].combine_first(df['Phases_2'])
#            df.drop(columns=["SMILES_2",'pred_Np_2'], inplace=True)
            df.drop(columns=["SMILES_2","Phases_2"], inplace=True)
            #    df1.sort_values(by=["ID"], ascending=True, inplace=True)

    df.to_csv(args.output)

#    i = 0
#     for index, row in df1.iterrows():
# #        print(row['ID'],df2.loc[i,'ID'],i)
#         while(row['ID']>df2.loc[i,'ID'] and i<len(df2)-1):
#             i += 1
#         if(row['ID']==df2.loc[i,'ID']):
# #            print("{},{},{},{},{},{},{},{},{},{},{}".format(row['ID'],row['Phases],row['Phases'],row['truth'],row['pred'],row['error'],row['ratio'],df2.loc[i,'truth'],df2.loc[i,'pred'],df2.loc[i,'error'],df2.loc[i,'ratio']))
#             print("{},{},{},{},{},{},{},{},{},{},{},{}".format(row['ID'],row['SMILES'],row['Phases'],row['truth'],row['pred'],row['error'],row['ratio'],df2.loc[i,'truth'],df2.loc[i,'pred'],df2.loc[i,'prob0'],df2.loc[i,'prob1'],df2.loc[i,'prob2']))

if __name__ == '__main__':
    main()
