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
    parser.add_argument('dataset1', help='Path to data file')
    parser.add_argument('dataset2', help='Path to data file')

    args = parser.parse_args()

    # read csv file
    df1 = pd.read_csv(args.dataset1, header=0)
    df1.sort_values(by=["ID"], ascending=True, inplace=True,)
    df1.reset_index(inplace=True)
    df2 = pd.read_csv(args.dataset2, header=0)
    df2.sort_values(by=["ID"], ascending=True, inplace=True)
    df2.reset_index(inplace=True)

    i = 0
#    print("ID,SMILES,Phases,truth1,pred1,error1,ratio1,truth2,pred2,error2,ratio2")
    print("ID,SMILES,Phases,truth,pred,error,ratio,truth,pred,prob0,prob1,prob2")
    for index, row in df1.iterrows():
#        print(row['ID'],df2.loc[i,'ID'],i)
        while(row['ID']>df2.loc[i,'ID'] and i<len(df2)-1):
            i += 1
        if(row['ID']==df2.loc[i,'ID']):
#            print("{},{},{},{},{},{},{},{},{},{},{}".format(row['ID'],row['SMILES'],row['Phases'],row['truth'],row['pred'],row['error'],row['ratio'],df2.loc[i,'truth'],df2.loc[i,'pred'],df2.loc[i,'error'],df2.loc[i,'ratio']))
            print("{},{},{},{},{},{},{},{},{},{},{},{}".format(row['ID'],row['SMILES'],row['Phases'],row['truth'],row['pred'],row['error'],row['ratio'],df2.loc[i,'truth'],df2.loc[i,'pred'],df2.loc[i,'prob0'],df2.loc[i,'prob1'],df2.loc[i,'prob2']))

if __name__ == '__main__':
    main()
