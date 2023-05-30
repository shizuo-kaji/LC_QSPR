# Quantitative Structure-Property Relationships for Liquid Crystals

This is a companion code for the paper
- Machine Learning Analysis for the Contribution of Molecular Fine Structures to the Symmetry of Liquid Crystals, by Yoshiaki Uchida, Shizuo Kaji, Naoto Nakano

## Requirements
First, to set up Python environment, install [Anaconda](https://www.anaconda.com/products/individual).

Install necessary libraries. 
The actual command for installation depends on the environment. 
For example, 

    > pip install iterative-stratification rdkit mordred pyarrow
    > pip install lightgbm
    > conda install pytorch torchvision torchaudio torcheval pytorch-cuda=11.8 -c pytorch -c nvidia
    > conda install pyg -c pyg
    > pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

## How to use
- Prepare a csv file containing at least the following two columns
    - ID: integer index to identify molecules uniquely
    - SMILES: SMILES one-line notation
    - Phases: Phase sequence in the format of LiqCryst

    a small sample file is included as *sample.csv*.

- Open the Jupyter notebook *LC_QSPR.ipynb* and follow the instructions.

- First, the database containing descriptors and phase transition temperatures should be created from the csv file above. 
Look at the *Compute descriptors from SMILES and parse Phase sequence* section.

- There are two main machine learning models: GBM and GNN. They can be tried separately.



