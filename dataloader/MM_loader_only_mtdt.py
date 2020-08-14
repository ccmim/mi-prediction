import numpy as np
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pdb


class MM(data.Dataset):
    """ Metadata Dataset.
        Args:
        dir_imgs (string): Root directory of dataset where images exist.
        is_train (bool): image for training or test
        ids_set (pandas class):
    """

    def __init__(self,
                dir_imgs,
                ids_set
                ):

        self.img_names = []
        self.mtdt = []
        self.labels = []
        # number fo participants
        self.num_parti = 0

        # scaler = MinMaxScaler()
        # mtdt_dataframe = ids_set[['sex', 'dbpa', 'sbpa', 'ss', 'ads', 'bmi', 'age', 'hba1c', 'chol', 'glucose']]
        # mtdt_scaled = pd.DataFrame(scaler.fit_transform(mtdt_dataframe), columns=mtdt_dataframe.columns)

        # Obtaining repeated ED and all sax images (labels)
        for idx, ID in enumerate(ids_set.values):
            self.num_parti = self.num_parti + 1

            # Reading all fundus images per patient
            imgs_per_id = glob.glob(dir_imgs + 'fundus/' + str(int(ID[0]))[0:2] + 'xxxxx/' + str(int(ID[0])) + '/*.png')

            # Taking only one image orientation -> left/right
            img_21015 = [j for j in imgs_per_id if '21016' in j]

            if len(img_21015) >= 1:
                imgs_per_id = img_21015[0]
                # Image names
                self.img_names.append(imgs_per_id.split('/')[-1][:-4].split('_')[0])

                # labels LVEDV_automatic[6], LVM_automatic[10]
                self.labels.append([ID[6], ID[10]])
                # mtd sex[24], dbpa[30], sbpa[31], ss[33], ads[34], bmi[36], age[38], hba1c[40], chol[41], glucose[43]
                self.mtdt.append([ID[24],  ID[30], ID[31], ID[33], ID[34], ID[36], ID[38], ID[40], ID[41], ID[43]])
                # self.mtdt.append([mtdt_scaled['sex'][idx], mtdt_scaled['dbpa'][idx], mtdt_scaled['sbpa'][idx], mtdt_scaled['ss'][idx],
                #                   mtdt_scaled['ads'][idx], mtdt_scaled['bmi'][idx], mtdt_scaled['age'][idx], mtdt_scaled['hba1c'][idx],
                #                   mtdt_scaled['chol'][idx], mtdt_scaled['glucose'][idx]
                #                   ])
            else:
                continue



    # Denotes the total number of samples
    def __len__(self):
        return len(self.img_names) # self.num_parti

    # This generates one sample of data
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (label, mtdt, img_name, index)
        """

        return torch.FloatTensor(self.labels[index]), torch.FloatTensor(self.mtdt[index]), self.img_names[index]


def mtdt_loader(batch_size,
              num_workers,
              shuffle,
              dir_imgs,
              ids_set):


    mtdt_dataset = MM(dir_imgs = dir_imgs,
                    ids_set = ids_set)

    print('Found ' + str(len(mtdt_dataset)) + ' fundus images')

    # Dataloader
    data_loader = torch.utils.data.DataLoader(mtdt_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader
