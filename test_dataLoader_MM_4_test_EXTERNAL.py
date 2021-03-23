from dataloader.MM_loader_4_test_EXTERNAL import MM_loader
import utils.io.image as io_func
from utils.sitk_np import np_to_sitk
import torchvision
import numpy as np
import pandas as pd
import torch
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing Loader')
    parser.add_argument('--dir_ids', type=str, default='./input_data_EXTERNAL/ids/ids_metadata_EXTERNAL.csv')
    parser.add_argument('--dir_dataset', type=str, default='./input_data_EXTERNAL/')
    args = parser.parse_args()

    # Reading the files that contains labels and names
    train_set = pd.read_csv(args.dir_ids, sep=',')


    data_loader = MM_loader(batch_size = 1,
                            fundus_img_size = 512,
                            num_workers = 4,
                            shuffle = True,
                            dir_imgs = args.dir_dataset,
                            ids_set = train_set
                            )

    iterator_loader = iter(data_loader)

    # Create folder where processed images are saved
    if not os.path.exists('./debug_loaders'):
        os.makedirs('./debug_loaders')

    for u in range(5):

        fundus, _, mtdt, img_names = next(iterator_loader)

        for n, img in enumerate(fundus):
            print('ID: ' + str(img_names[n]) + ' and Metadata ID: ' + str(mtdt[n].numpy()))
            # Saving image result
            # img_retina = (fundus[n] - torch.min(fundus[n]))/(torch.max(fundus[n]) - torch.min(fundus[n]))
            torchvision.utils.save_image(fundus[n], 'debug_loaders/fundus_generated_' + str(img_names[n]) + '.png')
