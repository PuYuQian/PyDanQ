# README for PyDanQ

DanQ is a hybrid convolutional and recurrent neural network model for predicting the function of DNA de novo from sequence. This implements by PyTorch again.

# Citing DanQ

Quang, D. and Xie, X. "DanQ: a hybrid convolutional and recurrent neural network for predicting the function of DNA sequences", NAR, 2015.

# INSTALL

Considering your ease of use, I have included the most recent version numbers of the software packages for the configuration that worked for me. For the record, I am using Ubuntu Linux 16.04 LTS with an NVIDIA Titan 1080Ti GPU and 32GB RAM.

## Required

- [Python] (<https://www.python.org>) (3.6.8). The easiest way to install Python and all of the necessary dependencies is to download and install [Anaconda] (https://www.anaconda.com/download/) (4.6.14).

- [PyTorch] (https://pytorch.org/) (1.0.1).

## Optional

- [CUDA] (https://developer.nvidia.com/cuda-80-download-archive) (8.0)

- [cuDNN] (https://developer.nvidia.com/rdp/cudnn-download) (7.1.3)

# USAGE

You need to first download the training, validation, and testing sets from DeepSEA. You can download the datasets from [here] (http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz). After you have extracted the contents of the tar.gz file, move the 3 .mat files into the data/ folder.

Because of my RAM limited, I firstly transform the train.mat file to 10 .pt files. You can get the 10 .pt files directly by using the **<u>*mat2pt_test.ipynb</u>*** for your convenience. (If you don't worry about this problem, you can fix the train-part code according to the valid-part code in DanQ_train.py file.)

Then you can train the model **<u>*DanQ_train.py</u>*** initially. Don't forget to install **visdom** and fix the **save_model_time** parameter according to your needs. Due to safety concerns, I set many model-saving checkpoints, you can fix it flexibility.(For your convenience, I've already uploaded the my bestmodel in the hyperlink, and I am grateful that if you can update it.)

When you have trained successfully, you can use **<u>*DanQ_test.ipynb</u>* ** to evaluate the model.Because of the  flexibility of the jupyter notebook, I integrate the pred, ROC/PR curve and aus file together.

You can generate the motif file(MEME) by using the **<u>*visualize_motif.ipynb</u>***. Before using this code, suggest you to read [this] (https://github.com/uci-cbcl/DanQ/issues/9) firstly.

You can generate the entropy matrix by using the **<u>*entropy_matrix.ipynb</u>***.When you get the entropy value,  you can use [seaborn] (http://seaborn.pydata.org/) to make the heatmap, which represents the information content of each letter.

## OPTIONAL

For convenience，you can download my trained [bestmodel] (https://pan.baidu.com/s/1L_dZZ3GNEyYethWblBMGzg) with the password 'x9xl' . 

# REFERENCE

> [DanQ] (https://github.com/uci-cbcl/DanQ/)