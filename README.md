
# Seeing your Heart through your Eyes: Predicting Myocardial Infarction using Retinal Images and Demographic Data

This is the project web for the study titled "Seeing your Heart through your Eyes: Predicting Myocardial Infarction using Retinal Images and Demographic Data". This is study is under review in the journal Nature Machine Intelligence.


## Contents

- [Abstract](#Abstract)
- [Proposed Method](#Proposed-Method)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Citation](#citation)

----------------

## Abstract:

Retinal images are routinely obtained in ophthalmologic practice to aid diagnosis and monitoring of both primary eye diseases and systemic conditions affecting the eye, such as diabetic retinopathy. Recent studies have shown that biomarkers on retinal images, e.g., retinal blood vessels density or tortuosity, are associated with cardiac function and may be used to identify patients at risk of coronary artery disease. In this work, we investigate the use of retinal images together with relevant patient metadata, to estimate left ventricular mass (LVM) and left ventricular end-diastolic volume (LVEDV), and subsequently, predict incident myocardial infarction. Retinal images, cardiac magnetic resonance (CMR) images and demographic data from **5663** participants in the UK Biobank cohort were used to train and validate a multi-channel variational autoencoder (mcVAE). Once trained, we were able to estimate LVM (4.4 (-32.30, 41.1) g) and LVEDV (3.02 (-53.45, 59.49) ml) based on the retinal images and demographic data. Prediction of individual risk of myocardial infarction (**AUC=0.80+/-0.02**) was performed on data for participants with only retinal images and demographic data that were not used to train our proposed method. Here we show that during routine retinal imaging, patients at high risk of future left ventricular hypertrophy and myocardial infarction could be identified.

----------------

## Proposed Method:

We used the UK Biobank dataset [UKB](https://www.ukbiobank.ac.uk/) (application # 11350 [PDF](https://www.ukbiobank.ac.uk/wp-content/uploads/2019/02/11350-Professor-Alejandro-Frangi.pdf)) to train and validate the proposed method. Our method is based on the work published in ICML by Antelmi et al. [PDF](http://proceedings.mlr.press/v97/antelmi19a/antelmi19a.pdf).

**Schema of the proposed method:** This system is composed of two main components, a multi-channel VAE and a deep regressor network. During Stage I, a joint latent space is created with two channels: Retinal and cardiac MR. Then, during Stage II a deep regressor is trained on the reconstructed CMR plus demographic data to estimate LVM and LVEDV. **Demographic data:** Summary of the subjects metadata used in this study to train (5097 participants) and test (566 participants) the proposed method. All continuous values are reported in mean and standard deviation (in parenthesis)  while categorical data are reported in percentage.


![Schema of the proposed method and Demographic data.](figure/retinal_CMR_mcVAE.jpg)


**Encoder and Decoder Architectures:** The following are the layers used for the decoder and decoder of both channels.

![Encoder and Decoder Architectures.](figure/EncDec_nets.png)

---------------------

## Repo Contents

- [dataloader](./dataloader): All dataloaders used in this study. Dataloaders to train and test the system with retinal images, cardiac MR and both image modalities.
- [mcvae](./mcvae): Multi-channel VAE model.
- [networks](./networks): Networks used as encoder and decoder in the mcVAE
- [statistical_analysis](./statistical_analysis): Jupyter Notebook used to perform the myocardial infarction predcition based on the cardiac indices and demographic data.
- [csvPreparation](./csvPreparation): Scripts used to prepare input csv files (This bit is based on the UK Biobank data). 


----------------

## System Requirements:

### OS Requirements

The package development version is tested on *Linux* operating systems. The developmental version of the package has been tested on the following systems:

Linux: CentOS Linux 7

### Package dependencies

We used Python 3.6.5. File **requirements.txt** contains the list of python libraries/packages used to train our approach.

### Weights

The weights of the trained models (~10GB) could be download from [OneDrive](https://leeds365-my.sharepoint.com/:f:/g/personal/scsad_leeds_ac_uk/EqL0iAuwIXFNt32KGADyv0EBStdbTXdVW8s5I5K1LY90aA?e=2fdChW).


## Installation Guide

### Testing

Users should first create a Python virtual environment and then install all the libraries listed in **requirements.txt** file using the command "pip install -r requirements.txt". This will take about 10 mins to install.

Secondly, users should download the weights inside the main folder.


Finally, users should run the script **mcVAE_4_test.py** to test on new retinal images. For testing, users may want to use the UK Biobank dataset or other publicly available datasets such as [HRF](https://www5.cs.fau.de/research/data/fundus-images/)


### Training

For training the system, users should load the cardiac MR, retinal images and demographic data inside the "./input_data" folder. Then, run the script "main_mcVAE.py" to train the mcVAE. 


### General Notes

1. Scripts **main_mcVAE.py**, **mcVAE_4_test.py** were used to train and test the Multi-channel VAE, respectively. (Stage I)
2. Script **main_deep_regressor.py** was used to train the Stage II. 
3. Scripts **main_only_fundus.py** and **main_only_mtdt.py** were used to analyse contribuntion made by retinal and demographic data respectively.



If you are having an issue that you believe to be tied to software versioning issues, please drop us an [Issue](https://github.com/diazandr3s/MI_pred_mcvae_ukbb/issues). 


 
----------------
## Citation:

Please cite the following paper if you use this code:

[1] **Diaz-Pinto A**, Attar R, Ravikumar N, Suinesiaputra A, Zhao Y, Levelt E, Dall'Armellina E, Lorenzi M, Gale C P, Gale R P, Plein S and Frangi A F. "Seeing your Heart through your Eyes: Predicting Myocardial Infarction using Retinal Images and Demographic Data". Nature Machine Intelligence. Under review. September 2020. [[PDF]](http://www.cistib.org/)



----------------

Update log:

- 20.08.14: Code released.
- 20.09.15: Network weights uploaded. Article under review.

