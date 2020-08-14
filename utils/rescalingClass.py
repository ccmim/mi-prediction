import pandas as pd
from sklearn import preprocessing
import numpy as np
import seaborn as sns

class scalingClass(object):

    def __init__(self, dir_data, dir_scaled_data, factor, save_scaled):


        self.factor = factor
        self.dir_data = dir_data
        self.dir_scaled_data = dir_scaled_data

        # Reading original data
        list_ids_values = pd.read_csv(self.dir_data)

        # Processing first chunk
        first_chunk = list_ids_values.LVEDV.values
        self.scaler_first_chunk = preprocessing.MinMaxScaler()
        # Fitting data on the scaler object
        self.scaled_first_chunk = self.scaler_first_chunk.fit_transform(first_chunk.reshape(-1, 1))
        self.scaled_first_chunk = self.scaled_first_chunk*self.factor

        # Processing second chunk
        second_chunk = list_ids_values.LVM.values
        self.scaler_second_chunk = preprocessing.MinMaxScaler()
        # Fitting data on the scaler object
        self.scaled_second_chunk = self.scaler_second_chunk.fit_transform(second_chunk.reshape(-1, 1))
        self.scaled_second_chunk = self.scaled_second_chunk*self.factor

        if save_scaled:
            # Modifying LVEDV and LVM values
            list_ids_values.LVEDV = self.scaled_first_chunk
            list_ids_values.LVM = self.scaled_second_chunk
            # Saving scaled values
            list_ids_values.set_index('ID').to_csv(self.dir_scaled_data, sep=',')


    def rescaled_preds(self, dir_preds):

        preds_root = dir_preds.rsplit('/', 1)[0] + '/'

        list_ids_preds = pd.read_csv(dir_preds)

        # Processing first chunk
        scaled_preds_first_chunk = list_ids_preds.LVEDV.values
        preds_first_chunk = self.scaler_first_chunk.inverse_transform(scaled_preds_first_chunk.reshape(-1, 1))
        preds_first_chunk = np.round(preds_first_chunk/self.factor)

        # Processing second chunk
        scaled_preds_second_chunk = list_ids_preds.LVM.values
        preds_second_chunk = self.scaler_second_chunk.inverse_transform(scaled_preds_second_chunk.reshape(-1, 1))
        preds_second_chunk = np.round(preds_second_chunk/self.factor)

        # Modifying LVEDV and LVM values
        list_ids_preds.LVEDV = preds_first_chunk
        list_ids_preds.LVM =   preds_second_chunk

        # Saving scaled values
        list_ids_preds.set_index('ID').to_csv(preds_root + 'preds_rescaled.csv', sep=',')
