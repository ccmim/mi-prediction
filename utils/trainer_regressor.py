import time
import torch
import numpy as np
import pandas as pd
from progress.bar import Bar


def train_step(train_loader, model, epoch, optimizer, criterion, scheduler, args):

    # switch to train mode
    model.train()
    epoch_loss = 0.0
    # loss_w = args.loss_w

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch+1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (_, recon_cmr, sax, labels, mtdt, img_names) in enumerate(train_loader):

        start_time = time.time()

        torch.set_grad_enabled(True)

        mtdt = mtdt.cuda()
        recon_cmr = recon_cmr.cuda()
        labels = labels.cuda()

        out_cmr = model(recon_cmr, mtdt)

        # print('\n This is out_cmr {} and labels {} '.format(str(out_cmr.cpu().detach().numpy()[0]), str(labels.cpu().detach().numpy()[0])))

        lossValue = criterion(out_cmr, labels)

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

    for step, (_, recon_cmr, _, labels, mtdt, img_names) in enumerate(val_loader):

        start_time = time.time()

        mtdt = mtdt.cuda()
        recon_cmr = recon_cmr.cuda()
        labels = labels.cuda()

        out_cmr = model(recon_cmr, mtdt)

        with torch.no_grad():
            lossValue = criterion(out_cmr, labels)
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
