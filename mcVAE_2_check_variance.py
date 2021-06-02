# Use this script to extract the variance

import numpy as np
import os
import torch
import pandas as pd
import argparse
import utils.io.image as io_func
from utils.sitk_np import np_to_sitk
from torchvision.utils import save_image
import pdb
import datetime
from shutil import copyfile
# from dataloader.MM_loader import MM_loader # Using both retinal and CMR images
from dataloader.MM_loader_4_test import MM_loader # using only retinal images
from networks.net_cmr_mtdt import net_cmr_mtdt
from mcvae import pytorch_modules, utilities
from utils.trainer_regressor import save_output



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test mcVAE/Deep Regressor')
    parser.add_argument('--n_channels', default=2, type=int) #  number of channels for MCVAE
    parser.add_argument('--lat_dim', default=2048, type=int)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--save_model', default=200, type=int) # save the model every x epochs
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--n_cpu', default=8, type=int)
    parser.add_argument('--dir_dataset', type=str, default='./input_data/')
    parser.add_argument('--dir_ids', type=str, default='./input_data/ids/automatic_LVM_LVEDV_mtdt_reduced.csv')
    parser.add_argument('--sax_img_size', type=list, default=[128, 128, 15])
    parser.add_argument('--fundus_img_size', type=int, default=128)
    parser.add_argument('--num_mtdt', type=int, default=10)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--dir_results', type=str, default='./results_test/')
    parser.add_argument('--dir_weights_mcvae', type=str, default='./results/2020-05-13_17-42-01_automatic_1800Epochs_reducedList/')
    parser.add_argument('--dir_weights_regressor', type=str, default='results_regressor/2021-04-05_01-06-29/')
    parser.add_argument('--model_name', type=str, default='net_cmr_mtdt')  # for deep regressor
    args = parser.parse_args()

    args.dir_weights_regressor = args.dir_weights_mcvae + args.dir_weights_regressor


    # Multi-channel VAE config
    init_dict = {
        'n_channels': args.n_channels,
        'lat_dim': args.lat_dim, # We fit args.lat_dim latent dimensions
        'n_feats': {'fundus': [3, args.ndf, args.fundus_img_size],
                    'cmr': [args.sax_img_size[2], args.ndf, args.sax_img_size[0]]
                    },
        'opt': args
    }

    print('\nTesting Mode. Loading IDs files \n')

    # Reading the files that contains labels and names.
    test_set = pd.read_csv(args.dir_ids, sep=',') # This is used to reconstructed both train and test set

    test_loader = MM_loader(batch_size = args.batch_size,
                               fundus_img_size = args.fundus_img_size,
                    			num_workers = args.n_cpu,
                                sax_img_size = args.sax_img_size,
        			            shuffle = False,
        			            dir_imgs = args.dir_dataset,
                                args = args,
                                ids_set = test_set
        			            )
    # Loading models
    print('Loading mcVAE model ...')
    # Creating model
    model_MM = pytorch_modules.MultiChannelSparseVAE(**init_dict)
    loaded_model = utilities.load_model(args.dir_weights_mcvae)
    model_MM.load_state_dict(loaded_model['state_dict'])

    print('Making predictions ...')

    laten_vars_fundus = []
    laten_vars_cmr = []
    img_names_4_linear_reg = []
    labels_4_linear_reg = []

    all_preds = torch.FloatTensor().cuda()
    image_names = []

    latent_vars_retina = np.zeros((5648, 2048)) # 2048 latent variables, 5648 images
    latent_vars_cmr = np.zeros((5648, 2048))

    for i, (fundus, sax, mtdt, img_names) in enumerate(test_loader):

        image_names.extend(img_names)

        fundus = fundus.cuda()
        sax = sax.cuda()
        mtdt = mtdt.cuda()

        print('Generating latent variables for image ...', i+1)
        # Getting predictions
        inputToLatent = model_MM.encode((fundus, sax))
        l_vars = model_MM.sample_from(inputToLatent)
        latent_vars_retina[i] = l_vars[0].cpu().detach().numpy()
        # latent_vars_cmr[i] = l_vars[1].cpu().detach().numpy()

    np.save('latent_vars_retina_using_retina_ONLY.npy', latent_vars_retina)
    # np.save('latent_vars_cmr_using_retina_CMR.npy', latent_vars_cmr)
