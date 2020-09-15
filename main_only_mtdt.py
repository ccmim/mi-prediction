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
from networks.net_mtdt import net_mtdt
from dataloader.MM_loader_only_mtdt import mtdt_loader
from progress.bar import Bar

warnings.filterwarnings("ignore", category=UserWarning)


def save_output(image_names, preds, args, save_file):

    label_list = ['LVEDV','LVM']
    label_list = label_list[:args.n_classes]
    n_class = args.n_classes
    np_preds = np.squeeze(preds.cpu().numpy())
    np_preds = np.round(np_preds, 4)
    result = {label_list[i]: np_preds[:, i] for i in range(n_class)}
    result['ID'] = image_names
    out_df = pd.DataFrame(result)
    name_older = ['ID']

    for i in range(n_class):
        name_older.append(label_list[i])
    out_df.to_csv(save_file, columns=name_older, index=False)


def train_step(train_loader, model, epoch, optimizer, criterion, scheduler, args):

    # switch to train mode
    model.train()
    epoch_loss = 0.0

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch+1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (labels, mtdt, img_names) in enumerate(train_loader):

        start_time = time.time()

        torch.set_grad_enabled(True)

        mtdt = mtdt.cuda()
        labels = labels.cuda()

        out_mtdt = model(mtdt)

        lossValue = criterion(out_mtdt, labels)

        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()
        scheduler.step() #  You should step scheduler after optimizer

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time

        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Error per batch: {loss:.4f} '
        bar.suffix = bar_str.format(step+1, iters_per_epoch, batch_time = batch_time*(iters_per_epoch-step)/60, loss = lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    print('\nAvg epoch error: {:.4f}'.format(epoch_loss))
    bar.finish()

    return epoch_loss




def validation_step(val_loader, model, criterion, args):

    # switch to train mode
    model.eval()
    epoch_loss = 0
    # loss_w = args.loss_w
    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)

    for step, (labels, mtdt, img_names) in enumerate(val_loader):

        start_time = time.time()

        mtdt = mtdt.cuda()
        labels = labels.cuda()

        out_mtdt = model(mtdt)

        with torch.no_grad():
            lossValue = criterion(out_mtdt, labels)
            epoch_loss += lossValue.item()

        end_time = time.time()

        # measure elapsed time
        batch_time = end_time - start_time
        bar_str = '{} / {} | Time: {batch_time:.2f} mins'
        bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    bar.finish()
    return epoch_loss



def train(model, train_loader, val_loader, args):

    best_metric = np.inf
    best_iter = 0

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999)) # , weight_decay=1e-5

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

        # Validating and saving model for each 5 epochs
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

    for epochID, (labels, mtdt, img_names) in enumerate(test_loader):

        mtdt = mtdt.cuda()
        labels = labels.cuda()

        IDs_imgs.extend(img_names)
        GT_labels.extend(labels.cpu().detach().numpy())

        begin_time = time.time()
        result_mtdt = model(mtdt)
        out_PREDS = torch.cat((out_PREDS, result_mtdt.data), 0)

        batch_time = time.time() - begin_time
        bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                               batch_time=batch_time * (iters_per_epoch - epochID) / 60)
        bar.next()
    bar.finish()
    return out_PREDS, IDs_imgs, GT_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Only Demographic')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dir_dataset', type=str, default='./input_data/')
    parser.add_argument('--dir_ids', type=str, default='./input_data/ids/automatic_LVM_LVEDV_mtdt.csv')
    parser.add_argument('--percentage', type=float, default=0.85)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_cpus', type=int, default=24)
    parser.add_argument('--num_mtdt', type=int, default=10)
    parser.add_argument('--save_model', type=str, default='net_mtdt') # This defines the model to use and the name of the weights file.
    parser.add_argument('--save_dir', type=str, default='results_only_mtdt/')
    parser.add_argument('--train', type=bool, default=False) # Change here to train or test the model. It'll take the latest trained model
    parser.add_argument('--results_dir', type=str, default='2020-05-24_19-01-07/') # Only change it when testing
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    # Seed
    np.random.seed(0)

    print('\nLoading model ...\n')

    model = globals()[args.save_model](args = args)
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


        train_loader = mtdt_loader(batch_size = args.batch_size,
                    			num_workers = args.n_cpus,
        			            shuffle = True,
        			            dir_imgs = args.dir_dataset,
                                ids_set = train_set
        			            )


        val_loader = mtdt_loader(batch_size = args.batch_size,
                    			num_workers = args.n_cpus,
        			            shuffle = True,
        			            dir_imgs = args.dir_dataset,
                                ids_set = val_set
        			            )


        args.save_dir = args.save_dir + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # Saving main file, dataloader, model and data division files
        copyfile('main_only_mtdt.py', args.save_dir + 'main_only_mtdt.py')
        copyfile('./dataloader/MM_loader_only_mtdt.py', args.save_dir + 'MM_loader_only_mtdt.py')
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

        args.save_dir = args.save_dir + args.results_dir

        # Reading the files that contains labels and names
        test_set = pd.read_csv(args.save_dir + 'test_set.csv', sep=',')

        test_loader = mtdt_loader(batch_size = args.batch_size,
                    			num_workers = args.n_cpus,
        			            shuffle = True,
        			            dir_imgs = args.dir_dataset,
                                ids_set = test_set
        			            )

        test_set = test_set[['ID', 'LVEDV_automatic', 'LVM_automatic']]
        test_set.to_csv(args.save_dir + 'test_set.csv', index=False)

        preds, image_names, GT_labels = test(model, test_loader, args)
        # Save result in a csv file
        pred_file_name =  args.save_dir + 'preds.csv'
        save_output(image_names, preds, args, save_file = pred_file_name)
