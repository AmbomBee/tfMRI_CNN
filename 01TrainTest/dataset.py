import torch
from torch.utils.data import Dataset
import csv
import numpy as np
import glob

class fMRIDataset(Dataset):
    """
    fMRI Data set
    """
    def __init__(self, label_dir, nii_dir, idc, transform=None):
        """
        Args:
            label_file (string): Path to the csv file with labels.
            nii_dir (string): directory of all the nii files (fMRI data).
            idc (list of ints): list of indices used for training, 
                validation or testing.
            transform (callable, optional): optional transform to be 
                applied on a sample.
            
        """
        def __labels__():
            """
            Returns a label dict with a list of label_IDs and the 
                matching list of labels.
            label_IDs are the corresponding filenames for the *.npy 
                files.
            """
            label_count = {'1': 0, '0': 0, '2': 0}
            label_dict = {'label_ID': [], 'label': [], 'trial_time': []}
            for name in self.label_filenames:
                with open(name, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file, delimiter=';')
                    line_count = 0
                    for row in csv_reader:
                        try:
                            label_dict['label_ID'].append(row['FileName'])
                        except KeyError as err:
                            print(err, flush=True)
                            print(name)
                            continue
                        label_dict['trial_time'].append(float(row['TrialTime']))
                        # Label 1 for success
                        if int(row['Success']) == 1:
                            label_dict['label'].append(1)
                            label_count['1'] += 1
                        # Label 0 for non success
                        elif int(row['Success']) == 0:
                            label_dict['label'].append(0)
                            label_count['0'] += 1
                        # Label 2 for reject bonus in bonus trial
                        # labeld as non success
                        elif int(row['Success']) == 2:
                            label_dict['label'].append(0)
                            label_count['2'] += 1
            print('Label count for whole: ', label_count, flush=True)
            return label_dict

        self.idc = idc
        self.label_filenames = np.asarray(glob.glob(label_dir 
                                                    + '*.csv'))
        self.nii_dir = nii_dir
        # only use the nii_filenames for the given indices
        self.nii_filenames = np.asarray(glob.glob(nii_dir 
                                                  + '*.npy'))[self.idc]
        self.transform = transform
        self.label_dict = __labels__()
                    
    def __len__(self):
        return len(self.nii_filenames)

    def __get_label__(self, img_name):
        """
        Returns the label to the matching label_ID through comparing 
        label_ID with filename of *.npy. 
        Catches the ValueError: filename not in label_ID list
        """
        try:
            # take img_name without '.nii' 
            # split and take only the filename without path to file
            filename = img_name[:-4].split('/')[-1]
            index = self.label_dict['label_ID'].index(filename)     
            return self.label_dict['label'][index]
        # handling exception for a filename not in the list of label_IDs
        except ValueError as err:
            print('Filename: ', filename, flush=True)
            #print(self.label_dict['label_ID'], flush=True)
            print("ValueError: {0}".format(err), flush=True)
            return None
    
    def __getitem__(self, idx):
        """
        supports the indexing of fMRIDataset
        """
        # label indices to trial indices (default len of patches = 200)
        img_name = self.nii_filenames[idx]
        fdata = np.load(img_name, mmap_mode='r')
        if fdata.shape[3] != 8:
            print('Length of {} does not match!'.format(img_name), flush=True)
        label = self.__get_label__(img_name)
        if label == None:
            print('label == None', flush=True)
        sample = {'fdata': fdata, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors
    """
    def __call__(self, sample):
        fdata, label = sample['fdata'], sample['label']
        # swap axis because
        # numpy image: H x W x D x C
        # torch image: C X H X W x D 
        fdata = fdata.transpose((3, 0, 1, 2))
        return {'fdata': torch.from_numpy(fdata),
                'label': torch.from_numpy(np.asarray(label))}
