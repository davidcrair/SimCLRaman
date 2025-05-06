# SimCLRaman - Supervised Contrastive Learning for Identification of Bacteria

### CPSC 552 Spring 2025 Final Project by David Crair

This repository contains the implementation of SimCLRaman, a supervised contrastive learning framework for classifying bacteria species using Raman spectroscopy data.

### Abstract
SimCLR and other contrastive learning frameworks have shown to be valuable in
learning useful representation of labeled visual data. Yet, contrastive learning on
1-dimensional data, specifically spectral data, has traditionally not been prevalent in
this domain. Raman spectroscopy is one special kind of 1D data in which machine
learning has recently risen to popularity. One somewhat popular classification
benchmark involves 30 different species of bacteria. This paper introduces a novel
application of contrastive learning to the problem of raman spectra classification,
adapting the SimCLR framework. We propose a model named SpectralNet which
includes supervised contrastive learning, 1D convolutional layers, and a non-local
attention layer. This model achieve a moderate improvement in classification
accuracy when compared to the original CNN model introduced with the dataset,
achieving 84.1±0.4% accuracy on the 30 species bacteria classification benchmark,
compared to the reference ResNet model accuracy of 82.2 ± 0.3%



### Installation
Use the requirements.txt file to install the required packages.

`pip install -r requirements.txt`  

If using the Yale HPC cluster, the following commands will create and activate a conda environment with the required packages:
```bash
conda create -yn simclraman python pip
conda activate simclraman
pip install -r requirements.txt
```


### Data
The data used in this project is from the original paper by Ho et al. (2019).\
Download the dataset from [here](https://www.dropbox.com/sh/gmgduvzyl5tken6/AABtSWXWPjoUBkKyC2e7Ag6Da?dl=0) and place it in the bacteria_data folder. This must be done manually, before running the code.\
The final folder should look like this:
```
bacteria_data/X_finetune.npy
bacteria_data/y_finetune.npy
bacteria_data/X_test.npy
bacteria_data/y_test.npy
bacteria_data/X_2018clinical.npy
bacteria_data/y_2018clinical.npy
bacteria_data/X_2019clinical.npy
bacteria_data/y_2019clinical.npy
```


### Running the Code
Set `PERFORM_PRETRAINING=True` to and `LOAD_PRETRAINED=FALSE` in the second code cell of the notebook to retrain the model.

Set `LOAD_PRETRAINED=TRUE` to load the pre-trained model and run the evaluation code.


### Pre-trained Models
We provide a pre-trained model for the encoder, which can be used for transfer learning.\
The model is saved in the saved_models folder as `finetuned_model_c50_s100.pt`.\
Refer to the above section for instructions on how to load the model and run the evaluation code.


### File Structure

```
.gitignore                           # standard gitignore file
README.md                            # this file
bacteria_data
    |--  DATA GOES HERE              # place the dataset here
contrastive_learning.ipynb           # Jupyter notebook for training and evaluation
contrastive_trainer.py               # code for training the encoder
datasets.py                          # code for loading the dataset (defines augmentations)
loss_functions.py                    # custom loss functions (SupConLoss and InfoNCE (not used))
saved_models
   |-- finetuned_model_c50_s100.pt   # pre-trained model
spectralnet.py                       # code for the encoder (SpectralNet)
utils.py                             # utility functions
```


### Citation
If you use this code, please cite:\
`Crair, D. (2025). SimCLRaman: Supervised Contrastive Learning for Identification of Bacteria. CPSC 552, Spring 2025.`


And the original dataset:\
`Ho, C.S., Jean, N., Hogan, C.A. et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Nat Commun 10, 4927 (2019). https://doi.org/10.1038/s41467-019-12898-9`
