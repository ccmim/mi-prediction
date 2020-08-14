import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
import cv2
import pandas as pd
import pdb

def scalRadius(img, scale):
    x = img[int(img.shape[0]/2),:,:].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    if r < 0.001: # This is for the very black images
        r = scale*2
    s = scale*1.0/r
    return cv2.resize(img, (0,0), fx=s, fy=s)


def load_preprocess_img(dir_img):
    scale = 300
    a = cv2.imread(dir_img)
    a = scalRadius(a,scale)
    a = cv2.addWeighted(a,4,cv2.GaussianBlur(a, (0,0), scale/30), -4, 128)
    b = np.zeros(a.shape)
    cv2.circle(b, (int(a.shape[1]/2),int(a.shape[0]/2)), int(scale*0.9), (1,1,1), -1, 8, 0)
    a = a*b + 128*(1-b)
    img = Image.fromarray(np.array(a, dtype=np.int8), "RGB")
    return img


class MM(data.Dataset):
    """ Multi-Modal Dataset.
        Args:
        dir_imgs (string): Root directory of dataset where images exist.
        transform_fundus: Tranformation applied to fundus images
        is_train (bool): image for training or test
	    fundus_img_size (int): Size for fundus images. i.e. 224
        ids_set (pandas class):
    """

    def __init__(self,
                dir_imgs,
                fundus_img_size,
                ids_set
                ):

        self.img_names = []
        self.fundus_img_size = fundus_img_size
        self.labels = []
        # fundus image paths
        self.path_imgs_fundus = []
        # number fo participants
        self.num_parti = 0


        # Obtaining repeated ED and all sax images (labels)
        for idx, ID in enumerate(ids_set.values):
            self.num_parti = self.num_parti + 1

            # Reading all fundus images per patient
            imgs_per_id = glob.glob(dir_imgs + 'fundus/' + str(int(ID[0]))[0:2] + 'xxxxx/' + str(int(ID[0])) + '/*.png')

            # Taking only one image orientation -> left/right
            img_21015 = [j for j in imgs_per_id if '21016' in j]

            if len(img_21015) >= 1:
                imgs_per_id = img_21015[0]
                # path for fundus images
                self.path_imgs_fundus.append(imgs_per_id)
                # Image names
                self.img_names.append(imgs_per_id.split('/')[-1][:-4].split('_')[0])

                # labels LVEDV_automatic[6], LVM_automatic[10]
                self.labels.append([ID[6], ID[10]])
            else:
                continue


        # Transform for fundus images
        self.transform_fundus = transforms.Compose([
                transforms.Resize((self.fundus_img_size, self.fundus_img_size)),
                transforms.ToTensor(),
            ])


    # Denotes the total number of samples
    def __len__(self):
        return len(self.path_imgs_fundus) # self.num_parti

    # This generates one sample of data
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (fundus, label, img_name, index)
        """

	    # Loading fundus image
        # preprocessing
        fundus = load_preprocess_img(self.path_imgs_fundus[index])
        # without preprocessing
        # fundus = Image.open(self.path_imgs_fundus[index]).convert('RGB')
        # resizing the images
        fundus_image = self.transform_fundus(fundus)
        # normalizing the images
        fundus_image = (fundus_image - torch.min(fundus_image))/(torch.max(fundus_image) - torch.min(fundus_image)) # Normalize between 0 and 1
        # fundus_image = 2.0*(fundus_image - torch.min(fundus_image))/(torch.max(fundus_image) - torch.min(fundus_image))-1.0  # Normalize between -1 and 1

        return fundus_image, torch.FloatTensor(self.labels[index]), self.img_names[index]


def fundus_loader(batch_size,
              fundus_img_size,
              num_workers,
              shuffle,
              dir_imgs,
              ids_set):


    ######### Create class Dataset MM ########
    fundus_dataset = MM(dir_imgs = dir_imgs,
                    fundus_img_size = fundus_img_size,
                    ids_set = ids_set)

    print('Found ' + str(len(fundus_dataset)) + ' fundus images')

    # Dataloader
    data_loader = torch.utils.data.DataLoader(fundus_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader
