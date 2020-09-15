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
from dataloader.MM_loader import MM_loader
from mcvae import pytorch_modules, utilities, preprocessing, plot, diagnostics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fundus/CMR Modalities')
    parser.add_argument('--n_channels', default=2, type=int) #  number of channels for MCVAE
    parser.add_argument('--lat_dim', default=2048, type=int)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--save_model', default=200, type=int) # save the model every x epochs
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_cpu', default=24, type=int)
    parser.add_argument('--dir_dataset', type=str, default='./input_data/')
    parser.add_argument('--dir_ids', type=str, default='./input_data/ids/manual_LVM_LVEDV_mtdt.csv')
    parser.add_argument('--sax_img_size', type=list, default=[128, 128, 15])
    parser.add_argument('--fundus_img_size', type=int, default=128)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--test_mode', type=bool, default=True)
    parser.add_argument('--save_test_imgs', type=bool, default=True)
    parser.add_argument('--dir_test_ids', type=str, default='2020-05-17_11-36-05/')
    parser.add_argument('--dir_results', type=str, default='./results/')
    parser.add_argument('--percentage', type=float, default=0.90)
    args = parser.parse_args()


    # Multi-channel VAE config
    init_dict = {
        'n_channels': args.n_channels,
        'lat_dim': args.lat_dim, # We fit args.lat_dim latent dimensions
        'n_feats': {'fundus': [3, args.ndf, args.fundus_img_size],
                    'cmr': [args.sax_img_size[2], args.ndf, args.sax_img_size[0]]
                    },
        'opt': args
    }


    if not args.test_mode:

        args.dir_results = args.dir_results + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'

        if not os.path.exists(args.dir_results):
            os.makedirs(args.dir_results)

        print('\nLoading IDs file\n')
        IDs = pd.read_csv(args.dir_ids, sep=',')

        # Dividing the number of images for training and test.
        IDs_copy = IDs.copy()
        train_set = IDs_copy.sample(frac = args.percentage, random_state=0)
        test_set = IDs_copy.drop(train_set.index)

        test_set.to_csv(args.dir_results + 'test_set.csv', index=False)
        train_set.to_csv(args.dir_results + 'train_set.csv', index=False)
        copyfile('main_mcVAE.py', args.dir_results +  'main_mcVAE.py')
        copyfile('./networks/VAE_net.py', args.dir_results + 'VAE_net.py')
        copyfile('./dataloader/MM_loader.py', args.dir_results + 'MM_loader.py')

        train_loader = MM_loader(batch_size = args.batch_size,
                                   fundus_img_size = args.fundus_img_size,
                        			num_workers = args.n_cpu,
                                    sax_img_size = args.sax_img_size,
            			            shuffle = True,
            			            dir_imgs = args.dir_dataset,
                                    args = args,
                                    ids_set = train_set
            			            )

        # Creating model
        model_MM = pytorch_modules.MultiChannelSparseVAE(**init_dict)

        model_MM.init_loss()
        model_MM.optimizer = torch.optim.Adam(model_MM.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
        # Optimizing model
        model_MM.optimize(epochs = args.epochs, data = train_loader)
        # Saving model
        utilities.save_model(model_MM, filename= args.dir_results + 'model_' + str(args.epochs) + '_epochs_dict')

        # Creating plots
        print('Significant dimensions: ', model_MM.dropout.cpu().detach().numpy())
        significant_dim = np.where(model_MM.dropout.cpu().detach().numpy()<0.5)[1]

        # Plotting model convergence
        diagnostics.plot_loss(model_MM, save_fig=True, path_plot = args.dir_results)

    else:

        print('\nTesting Mode. Loading IDs files \n')

        # Reading the files that contains labels and names.
        # test_set = pd.read_csv(args.dir_results + args.dir_test_ids + 'train_set.csv', sep=',')
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
        # Loading model
        print('Loading model ...')
        # Creating model
        model_MM = pytorch_modules.MultiChannelSparseVAE(**init_dict)
        loaded_model = utilities.load_model(args.dir_results + args.dir_test_ids)
        model_MM.load_state_dict(loaded_model['state_dict'])

        print('Making predictions ...')

        laten_vars_fundus = []
        laten_vars_cmr = []
        img_names_4_linear_reg = []
        labels_4_linear_reg = []

        for i, (fundus, sax, mtdt, img_names) in enumerate(test_loader):

            fundus = fundus.cuda()
            sax = sax.cuda()

            print('Batch: ' + str(i))
            # Getting predictions
            inputToLatent = model_MM.encode((fundus, sax))
            latent_vars = model_MM.sample_from(inputToLatent)
            predictions = model_MM.decode(latent_vars)

            ############# For linear regression ############
            # Running for the batch size
            for l in range(len(img_names)):
                # concatenating latent variables and demographic
                laten_vars_fundus.append(np.concatenate([latent_vars[0].cpu().detach().numpy()[l], mtdt.cpu().detach().numpy()[l][2:]]))
                laten_vars_cmr.append(np.concatenate([latent_vars[1].cpu().detach().numpy()[l], mtdt.cpu().detach().numpy()[l][2:]]))
                labels_4_linear_reg.append(mtdt.cpu().detach().numpy()[l][:2])
                img_names_4_linear_reg.append(int(img_names[l].split('_')[0]))
            ###################################################

            # Dir for the generated data
            gen_dir = args.dir_results + args.dir_test_ids + 'gen_data/'

            if not os.path.exists(gen_dir):
                os.makedirs(gen_dir)

            for d in range(len(img_names)):

                # Saving image result
                if args.save_test_imgs:
                    n = min(fundus.size(0), 8)
                    comparison = torch.cat([fundus[:n], predictions[0][0].loc[:n]])
                    # save_image(comparison.cpu(), gen_dir + 'fundus_generated_{}'.format(img_names[:n]) + '.png', nrow=n)
                    # save_image(predictions[0][0].loc[d], gen_dir + 'fundus_generated_' + img_names[d] + '.png')
                    for idx in range(fundus.size(0)):
                        name_cmr = img_names[idx].split('_')[0]
                        io_func.write(np_to_sitk(predictions[0][1].loc.cpu().detach().numpy()[idx]), gen_dir + 'reconstructed_' + name_cmr + '.vtk')

        from utils.linear_reg import linear_reg
        linear_reg(laten_vars_fundus, labels_4_linear_reg, img_names_4_linear_reg, 'fundus', args)
        # linear_reg(laten_vars_cmr, labels_4_linear_reg, img_names_4_linear_reg, 'cmr', args)
