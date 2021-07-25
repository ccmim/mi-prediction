
# Predicting Infarction through your Retinal Scans and Basic Demographics

This is the project web for the study titled "Predicting Infarction through your Retinal Scans and Basic Demographics". This study is under review in the journal Nature Machine Intelligence.


----------------

## System Requirements:

### OS Requirements

The package development version is tested on *Linux* operating systems. The developmental version of the package has been tested on the following systems:

Linux: CentOS Linux 7 and Ubuntu 

### Package dependencies

We used Python 3.6.5. File **requirements.txt** contains the list of python libraries/packages used to train our approach.

### Weights

The weights of the trained models (~5GB) could be download from this [link](https://emckclac-my.sharepoint.com/:f:/g/personal/k2039747_kcl_ac_uk/EqjWo8c37A1LvuVGJcF9XhwBoh5d-7Sy-vPsewBaA3jkeQ?e=NtNTzW).


### Metadata coding

For 'ss' (smoking status) and 'ads' alcohol drinker status:

- -1 Not available or Prefer not to answer
- 0 Never
- 1 Previous
- 2 Current


bmi (body mass index) -> kg/m2

dbpa (diastolic blood pressure) -> mmHg

sbpa (systolic blood pressure) -> mmHg

sex -> 0 for Female and 1 for Male

## Steps for external validation


- As the first step, users have two options: (1) create a Python virtual environment and then install all the libraries listed in **requirements.txt** file using the command "pip install -r requirements.txt". This will take about 10 mins to install. (2) Use the Docker recipe to create a Docker container.

- Secondly, modify the dataloader file ('dataloader/MM_loader_4_test_EXTERNAL.py') according to the data organization you have. Then, create the input file (.csv file) following the example located in './input_data_EXTERNAL/ids/ids_metadata_EXTERNAL.csv'. 

- For this toy example, I used a small part ([100 images](https://drive.google.com/file/d/1zRBjk5cX7JLUXv4cFPdNMD4W5_vNga6M/view)) of the DR Kaggle dataset and I randomly draw metadata values for each subject: ID,sex,dbpa,sbpa,ss,ads,bmi,age


You can use the script **test_dataLoader_MM_4_test_EXTERNAL.py** to test your dataloader.


I uploaded the weights of a system trained with the following metadata: gender ('sex'), [smoking status](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20116) ('ss'), [drinking status](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=1558) ('ads'), body mass index ('bmi'), age ('age'), and diastolic blood pressure ('dbpa'), systolic blood pressure ('sbpa').


- Thirdly, you should download the [weights](https://emckclac-my.sharepoint.com/:f:/g/personal/k2039747_kcl_ac_uk/EqjWo8c37A1LvuVGJcF9XhwBoh5d-7Sy-vPsewBaA3jkeQ?e=NtNTzW) inside a folder called 'results'. This means, the weights should be located inside **MI_pred_mcvae_ukbb/results/**


- Finally, you should run the script **main_EXTERNAL.py** to test on the retinal images.

- The output files are located in folder **MI_pred_mcvae_ukbb/results_test/**



If you are having an issue that you believe to be tied to software versioning issues, please drop us an [Issue](https://github.com/diazandr3s/MI_pred_mcvae_ukbb/issues).

 
----------------

Update log (Year.Month.Day):

- 20.08.14: Code released.
- 20.09.15: Network weights uploaded. Article under review.
- 21.22.03: Created a branch for external validation
- 21.08.04: Updated weights and main file for 7 metadata variables - External validation
- 21.02.06: Upload all scripts used to address reviewers comments

