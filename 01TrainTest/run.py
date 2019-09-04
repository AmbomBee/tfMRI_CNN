from network import Net
import utils
import dataset
import test
import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import numpy as np
import glob
import time
from tqdm.auto import tqdm

def train(data, optimizer, net, criterion, device):
    # get the inputs
    inputs = data['fdata'].to(device)
    labels = data['label'].type(torch.LongTensor).to(device)
    # Clear all accumulated gradients
    optimizer.zero_grad()
    # Predict classes using inputs from the train set
    outputs = net(inputs)
    # Compute the loss based on the oututs and actual labels
    loss = criterion(outputs, labels)
    # Backpropagate the loss
    loss.backward()
    # Adjust parameters according to the computed 
    # gradients -- weight update
    optimizer.step()
    # Argmax of outputs
    _, predictions = torch.max(outputs.data, 1)
    return (predictions.cpu().numpy(), labels.cpu().numpy(), loss.item())
    
def val(data, net, criterion, device):
    # get the inputs
    inputs = data['fdata'].to(device)
    labels = data['label'].type(torch.LongTensor).to(device)
    # Predict classes using inputs from the val set
    outputs = net(inputs)
    # Compute the loss based on the outputs and actual labels
    loss = criterion(outputs, labels)
    # Argmax of outputs
    _, predictions = torch.max(outputs.data, 1)
    return (predictions.cpu().numpy(), labels.cpu().numpy(), loss.item())

def run(path_to_net, label_dir, nii_dir, plotter, batch_size=32, 
        test_split=0.3, random_state=666, epochs=8, 
        learning_rate=0.0001, momentum=0.9, num_folds=5):
    """
    Applies training and validation on the network 
    """
    print('Setting started', flush=True)
    nii_filenames = np.asarray(glob.glob(nii_dir + '/*.npy'))
    print('Number of files: ', len(nii_filenames), flush=True)
    # Creating data indices
    dataset_size = len(nii_filenames)
    indices = list(range(dataset_size))
    test_indices, trainset_indices = utils.get_test_indices(indices, 
                                                     test_split)
    # kfold index generator
    for cv_num, (train_idx, val_idx) in enumerate(utils.get_train_cv_indices(trainset_indices, 
                                                         num_folds, 
                                                         random_state)):
        # take from trainset_indices the kfold generated ones
        train_indices = np.asarray(trainset_indices)[np.asarray(train_idx)]
        val_indices = np.asarray(trainset_indices)[np.asarray(val_idx)]
        print('cv cycle number: ', cv_num, flush=True)
        net = Net()
        device = torch.device('cuda:0' 
                        if torch.cuda.is_available() else 'cpu')
        num_GPU = torch.cuda.device_count()
        print('Device: ', device, flush=True)
        if  num_GPU > 1:
            print('Let us use', num_GPU, 'GPUs!', flush=True)
            net = nn.DataParallel(net)
        net.to(device)
        # weigh the loss with the size of classes
        # class 0: 3268
        # class 1: 60248
        weight = torch.tensor([1./3268., 1./60248.]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, threshold=1e-6, patience=0, verbose=True)
        fMRI_dataset_train = dataset.fMRIDataset(label_dir, nii_dir, 
                                    train_indices, transform=dataset.ToTensor())
        fMRI_dataset_val = dataset.fMRIDataset(label_dir, nii_dir,
                                    val_indices, transform=dataset.ToTensor())
        datalengths = {'train': len(fMRI_dataset_train), 
                       'val': len(fMRI_dataset_val)}
        dataloaders = {'train': utils.get_dataloader(fMRI_dataset_train, 
                                               batch_size, num_GPU),
                      'val': utils.get_dataloader(fMRI_dataset_val, 
                                            batch_size, num_GPU)}
        print('Train set length {}, Val set length {}: '.format(datalengths['train'],
                                                                datalengths['val']))
        # Setup metrics
        running_metrics_val = metrics.BinaryClassificationMeter()
        running_metrics_train = metrics.BinaryClassificationMeter()
        val_loss_meter = metrics.averageLossMeter()
        train_loss_meter = metrics.averageLossMeter()
        # Track iteration number over epochs for plotter
        itr = 0
        # Track lowest loss over epochs for saving network
        lowest_loss = 100000
        for epoch in tqdm(range(epochs), desc='Epochs'):
            print('Epoch: ', epoch+1, flush=True)
            print('Phase: train', flush=True)
            phase = 'train'
            # Set model to training mode
            net.train(True)  
            # Iterate over data.
            for i, data in tqdm(enumerate(dataloaders[phase]), desc='Dataiteration_train'):
                train_pred, train_labels, train_loss = train(data, optimizer, net, criterion,
                                                             device)
                running_metrics_train.update(train_pred, train_labels)
                train_loss_meter.update(train_loss, n=1)
                if (i + 1) % 10 == 0:
                    print('Number of Iteration [{}/{}]'.format(i+1, 
                          int(datalengths[phase]/batch_size)), flush=True)
                    itr += 1
                    score = running_metrics_train.get_scores()
                    for k, v in score.items():
                        plotter.plot(k, 'itr', phase, k, itr, v)
                        print(k, v, flush=True)
                    print('Loss Train', train_loss_meter.avg, flush=True)
                    plotter.plot('Loss', 'itr', phase, 'Loss Train', 
                                 itr, train_loss_meter.avg)
                    utils.save_scores(running_metrics_train.get_history(), phase, cv_num)
                    utils.save_loss(train_loss_meter.get_history(), phase, cv_num)
            print('Phase: val', flush=True)
            phase = 'val'
            # Set model to validation mode
            net.train(False) 
            with torch.no_grad():
                for i, data in tqdm(enumerate(dataloaders[phase]), desc='Dataiteration_val'):
                    val_pred, val_labels, val_loss = val(data,net, 
                                                         criterion, device)
                    running_metrics_val.update(val_pred, val_labels)
                    val_loss_meter.update(val_loss, n=1)
                    if (i + 1) % 10 == 0:
                        print('Number of Iteration [{}/{}]'.format(i+1, 
                              int(datalengths[phase]/batch_size)), flush=True)
                    utils.save_scores(running_metrics_val.get_history(), phase, cv_num)
                    utils.save_loss(val_loss_meter.get_history(), phase, cv_num)
                    if val_loss_meter.avg < lowest_loss:
                        lowest_loss = val_loss_meter.avg
                        utils.save_net(path_to_net, batch_size, epoch, cv_num, 
                                       train_indices, val_indices, 
                                       test_indices, net, optimizer, criterion, iter_num=i)
                # Plot validation metrics and loss at the end of the val phase
                score = running_metrics_val.get_scores()
                for k, v in score.items():
                    plotter.plot(k, 'itr', phase, k, itr, v)
                    print(k, v, flush=True)
                print('Loss Val', val_loss_meter.avg, flush=True)
                plotter.plot('Loss', 'itr', phase, 'Loss Val', 
                             itr, val_loss_meter.avg)

            print ('Epoch [{}/{}], Train_loss: {:.4f}, Train_bacc: {:.2f}' 
                           .format(epoch+1, epochs, train_loss_meter.avg, 
                                   running_metrics_train.bacc), flush=True)
            print ('Epoch [{}/{}], Val_loss: {:.4f}, Val_bacc: {:.2f}' 
                           .format(epoch+1, epochs, val_loss_meter.avg, 
                                   running_metrics_val.bacc), flush=True)
            # Call the learning rate adjustment function after every epoch
            scheduler.step(train_loss_meter.avg)
        # Save net after every cross validation cycle
        utils.save_net(path_to_net, batch_size, epochs, cv_num, 
                       train_indices, val_indices, test_indices, net, 
                       optimizer, criterion)
