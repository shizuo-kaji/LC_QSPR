# Quantitative Structure-Property Relationships for Liquid Crystals

This is a companion code for the paper
- Machine Learning Analysis for the Contribution of Molecular Fine Structures to the Symmetry of Liquid Crystals, by Yoshiaki Uchida, Shizuo Kaji, Naoto Nakano

## Requirements
First, to set up Python environment, install [Anaconda](https://www.anaconda.com/products/individual).

Install [mordred](https://github.com/mordred-descriptor/mordred) by

    > conda install -c conda-forge rdkit
    > pip install mordred

Currently, mordred requires an older version of networkx:

    > conda install networkx=2.3

Install Chainer and chainer-chemistry

    > pip install chainer, chainer-chemistry


If you are on Windows, install Perl by

    > conda install perl

## How to use
### create descriptor table
- Prepare data files (sample files are found under the directory named *data*, but they are too small for actual use)
    - *Smiles_sample.txt*: one SMILES per line. e.g., 

    Smiles	C1C=CC=CC=1

    - *raw_sample.txt*: text data from the database
    - (optional) *TruthTable_sample.txt*: each line is either "true" or "false". Only those lines flagged as "true" will be computed.

- (On Windows) open compute_desc.bat and edit variables +dpath+ and +num+ appropriately,
and execute it. Wait for some time and you will obtain *desc_sample.csv*.

- (On Unix/Mac)
    - create an smi file combining the information of the above files (TruthTable_sample.txt is optional)
        > perl raw2phase.pl smiles_sample.txt raw_sample.txt TruthTable_sample.txt > sample.smi

     - compute descriptors using mordred (takes time, and may produce some ignorable errors)

        > python -m mordred -p 1 sample.smi -o temp.csv

     - add a header to test.csv so that the resulting file desc_sample.csv can be loaded to the R code.

        > perl add-header.pl temp.csv > desc_sample.csv



### Prediction by R (lightGBM etc)

- Open the R Notebook *qspr.Rmd* in RStudio and load *desc_sample.csv*. 
The R Notebook contains several regression and classification analysis.

### Prediction by Chainer Chemistry
- Prepare training and test datasets in the csv format. The file should contain SMILES and target value to be predicted. No descriptor information is needed. 

    > python chainerChem.py -t train.csv -v test.csv -m relgcn -u 64 -c 8 -b 16 -l ColumnNameOfTargetValue -e 400

Look at the help message

    > python chainerChem -h


### Prediction by MLP
- To regress certain value or to classify using a neural network

    > python MLP.py -l 5 desc_sample.csv

(-l 5) means the regression is for Melting temperature, which is recorded at the 5th column of desc_sample.csv.
(Here, we count the columns from 0.)

Look at the help message

    > python MLP.py -h

for the list of hyper-parameters and how to use it for classification.
Note that tuning and preprocessing is essential for a reasonable performance.
Keep in mind that plain use of the above will rarely produce good results.


