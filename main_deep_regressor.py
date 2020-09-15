import warnings
import os
import numpy as np
from numpy import argmax
import argparse
import torch
import torch.nn as nn
from torch.nn import init
import time
import pandas as pd
import glob
import datetime
from shutil import copyfile
from torch.optim import Adam, Adadelta, lr_scheduler
from tqdm import tqdm
from utils.metric import compute_metric
from networks.net_cmr_mtdt import net_cmr_mtdt
from dataloader.MM_loader_reg import MM_loader
from utils.trainer_regressor import train_step, validation_step, save_output
from progress.bar import Bar

warnings.filterwarnings("ignore", category=UserWarning)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def train(model, train_loader, val_loader, args):

    best_metric = np.inf
    best_iter = 0

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)

    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()

    # More schdulers https://pytorch.org/docs/stable/optim.html
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs/1)])


    # Counting the number of parameters
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    losses = []

    for epoch in range(0, args.epochs):

        epoch_loss = train_step(train_loader, model, epoch, optimizer, criterion, scheduler, args)

        losses.append(epoch_loss)

        # Validating and saving model for each 1 epochs
        if (epoch % 5) == 0:
            validation_loss = validation_step(val_loader, model, criterion, args)
            print('Current Error: {}| Best Error: {} at epoch: {}'.format(validation_loss, best_metric, best_iter))

            # save model
            if best_metric > validation_loss:
                best_metric = validation_loss
                best_iter = epoch
            model_save_file = os.path.join(args.save_dir, args.save_model + '.tar')
            torch.save({'state_dict': model.state_dict(), 'best_error': best_metric}, model_save_file)
            print('Model saved to %s' % model_save_file)

    return losses

def test(model, test_loader, args):

    IDs_imgs = []
    GT_labels = []

    print('\nLoading trained model ...\n')

    if args.save_model is not None:
        loaded_model = torch.load(os.path.join(args.save_dir, args.save_model + '.tar'))
        model.load_state_dict(loaded_model['state_dict'])

    # Testing
    out_PREDS = torch.FloatTensor().cuda()
    model.eval()
    iters_per_epoch = len(test_loader)
    bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
    bar.check_tty = False

    for epochID, (_, recon_cmr, _, labels, mtdt, img_names) in enumerate(test_loader):

        mtdt = mtdt.cuda()
        recon_cmr = recon_cmr.cuda()
        labels = labels.cuda()

        IDs_imgs.extend(img_names)
        GT_labels.extend(labels.cpu().detach().numpy())

        begin_time = time.time()
        result_cmr = model(recon_cmr, mtdt)
        out_PREDS = torch.cat((out_PREDS, result_cmr.data), 0)

        batch_time = time.time() - begin_time
        bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                               batch_time=batch_time * (iters_per_epoch - epochID) / 60)
        bar.next()
    bar.finish()
    return out_PREDS, IDs_imgs, GT_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CMR/Demographic')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dir_dataset', type=str, default='./input_data/')
    parser.add_argument('--dir_mcvae_res', type=str, default='./results/2020-05-10_19-44-26_automatic/')
    parser.add_argument('--percentage', type=float, default=0.99)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_cpus', type=int, default=12)
    parser.add_argument('--sax_img_size', type=list, default=[128, 128, 15])
    parser.add_argument('--num_mtdt', type=int, default=10)
    parser.add_argument('--save_model', type=str, default='net_cmr_mtdt') # This defines the model to use and the name of the weights file. It can be mmf_net_v4 and mmf_net_v6
    parser.add_argument('--train', type=bool, default=False) # Change here to train or test the model. It'll take the latest trained model
    parser.add_argument('--results_dir', type=str, default='2020-05-12_00-28-28/') # Only change it when testing
    args = parser.parse_args()

    args.dir_recon_cmr = args.dir_mcvae_res + 'gen_data/'
    args.dir_ids = args.dir_mcvae_res + 'train_set.csv'
    args.save_dir = args.dir_mcvae_res + 'results_regressor/'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    # Seed
    np.random.seed(0)

    print('\nLoading model ...\n')

    model = globals()[args.save_model](args = args)
    # model.apply(weights_init)
    model = model.to(device)
    # print(model)


    if args.train:

        print('\nLoading IDs files \n')

        # Reading the files that contains labels and names
        IDs = pd.read_csv(args.dir_ids, sep=',')

        # Dividing the number of images for training and test.
        IDs_copy = IDs.copy()
        train_set = IDs_copy.sample(frac = args.percentage, random_state=0)
        val_set = IDs_copy.drop(train_set.index)


        train_loader = MM_loader(batch_size = args.batch_size,
                                   fundus_img_size = 256, # We can remove this bit
                        			num_workers = args.n_cpus,
                                    sax_img_size = args.sax_img_size,
            			            shuffle = True,
            			            dir_imgs = args.dir_dataset,
                                    dir_recon_cmr = args.dir_recon_cmr,
                                    ids_set = train_set
            			            )


        val_loader = MM_loader(batch_size = args.batch_size,
                                   fundus_img_size = 256, # We can remove this bit
                        			num_workers = args.n_cpus,
                                    sax_img_size = args.sax_img_size,
            			            shuffle = True,
            			            dir_imgs = args.dir_dataset,
                                    dir_recon_cmr = args.dir_recon_cmr,
                                    ids_set = val_set
            			            )


        args.save_dir = args.save_dir + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        # Saving main file, dataloader, model and data division files
        copyfile('main_deep_regressor.py', args.save_dir + 'main_deep_regressor.py')
        copyfile('./dataloader/MM_loader_reg.py', args.save_dir + 'MM_loader_reg.py')
        copyfile('./networks/' + args.save_model + '.py', args.save_dir + args.save_model + '.py')
        val_set.to_csv(args.save_dir + 'test_set.csv', index=False)
        train_set.to_csv(args.save_dir + 'train_set.csv', index=False)
        losses = train(model, train_loader, val_loader, args)
        # Saving epoch losses
        out_df = pd.DataFrame(losses)
        out_df.to_csv(args.save_dir + 'epoch_errors.csv', header=False, index=False)
        preds, image_names, GT_labels = test(model, val_loader, args)
        # Save result in a csv file
        pred_file_name = args.save_dir + 'preds.csv'
        save_output(image_names, preds, args, save_file = pred_file_name)


    else:

        print('\nTesting Mode. Loading IDs files \n')

        # Reading the files that contains labels and names
        test_set = pd.read_csv(args.dir_mcvae_res + 'test_set.csv', sep=',')

        test_loader = MM_loader(batch_size = args.batch_size,
                                   fundus_img_size = 256, # We can remove this bit
                        			num_workers = args.n_cpus,
                                    sax_img_size = args.sax_img_size,
            			            shuffle = False,
            			            dir_imgs = args.dir_dataset,
                                    dir_recon_cmr = args.dir_recon_cmr,
                                    ids_set = test_set
            			            )
        args.save_dir = args.save_dir + args.results_dir

        if len(test_set.columns) > 41: # For automatic values
            test_set = test_set[['ID', 'LVEDV_automatic', 'LVM_automatic']]
        else: # Manual values
            test_set = test_set[['ID', 'LVEDV', 'LVM']]

        test_set.to_csv(args.save_dir + 'test_set.csv', index=False)

        preds, image_names, GT_labels = test(model, test_loader, args)
        # Save result in a csv file
        pred_file_name =  args.save_dir + 'preds.csv'
        save_output(image_names, preds, args, save_file = pred_file_name)
