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
from dataloader.MM_loader_4_test import MM_loader
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
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_cpu', default=24, type=int)
    parser.add_argument('--dir_dataset', type=str, default='../input_data/')
    parser.add_argument('--dir_ids', type=str, default='./input_data/ids/non_stroke_MI_ids_mtdt.csv') # non_stroke_MI_ids_mtdt # ids_MI_4_test_mtdt
    parser.add_argument('--sax_img_size', type=list, default=[128, 128, 15])
    parser.add_argument('--fundus_img_size', type=int, default=128)
    parser.add_argument('--num_mtdt', type=int, default=10)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--dir_results', type=str, default='./results_test/')
    parser.add_argument('--dir_weights_mcvae', type=str, default='./results/2020-05-10_19-44-26_automatic/')
    parser.add_argument('--dir_weights_regressor', type=str, default='results_regressor/2020-05-12_00-28-28/')
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

    print('Loading deep regressor model ...')

    model_reg = globals()[args.model_name](args = args)
    loaded_model = torch.load(os.path.join(args.dir_weights_regressor, args.model_name + '.tar'))
    model_reg.load_state_dict(loaded_model['state_dict'])
    model_reg = model_reg.cuda()
    model_reg.eval()

    print('Making predictions ...')

    laten_vars_fundus = []
    laten_vars_cmr = []
    img_names_4_linear_reg = []
    labels_4_linear_reg = []

    if not os.path.exists(args.dir_results):
        os.makedirs(args.dir_results)

    all_preds = torch.FloatTensor().cuda()
    image_names = []

    for i, (fundus, sax, mtdt, img_names) in enumerate(test_loader):

        image_names.extend(img_names)

        fundus = fundus.cuda()
        sax = sax.cuda()
        mtdt = mtdt.cuda()

        print('Generating CMR ...')
        # Getting predictions
        inputToLatent = model_MM.encode((fundus, sax))
        latent_vars = model_MM.sample_from(inputToLatent)
        predictions = model_MM.decode(latent_vars)

        print('Estimating cardiac indices ...')
        recons_cmr = predictions[0][1].loc
        # recons_cmr = (recons_cmr - torch.min(recons_cmr))/(torch.max(recons_cmr) - torch.min(recons_cmr)) # Normalize between 0 and 1
        preds = model_reg(recons_cmr, mtdt)
        # print(preds.cpu().detach().numpy())
        all_preds = torch.cat((all_preds, preds.data), 0)

        # Saving the reconstructed CMR
        # for d in range(len(img_names)):
        #     # Saving image result
        #     for idx in range(fundus.size(0)):
        #         name_cmr = img_names[idx].split('_')[0]
        #         io_func.write(np_to_sitk(predictions[0][1].loc.cpu().detach().numpy()[idx]), args.dir_results + 'reconstructed_' + name_cmr + '.vtk')

    pred_file_name =  args.dir_results + args.dir_ids.split('/')[-1][:-4]  + '_preds.csv'
    save_output(image_names, all_preds, args, save_file = pred_file_name)
