import pickle

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from enums import FeatureType


class MELDDataset(Dataset):
    """
    A custom Dataset class for the MELD dataset.
    Args:
        path (str): Path to the dataset file.
        n_classes (int): Number of classes (3 or 7).
        feature_types (FeatureType): Types of features to be used (TEXT, AUDIO, VISUAL, etc.).
        train (bool, optional): If True, use training data; otherwise, use test data. Default is True.
    Attributes:
        videoIDs (list): List of video IDs.
        videoSpeakers (list): List of video speakers.
        videoLabels (list): List of video labels.
        videoText (list): List of video text features.
        videoAudio (list): List of video audio features.
        videoVisual (list): List of video visual features.
        videoSentence (list): List of video sentences.
        trainVid (list): List of training video IDs.
        testVid (list): List of test video IDs.
        keys (list): List of keys for the current dataset split (train or test).
        len (int): Length of the dataset.
        feature_types (FeatureType): Types of features to be used.
    Methods:
        __getitem__(index):
            Returns the data and label for a given index.
        __len__():
            Returns the length of the dataset.
        collate_fn(data):
            Custom collate function for DataLoader to handle variable-length sequences.
    """

    def __init__(self, path, n_classes, feature_types, train=True):
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText, \
                self.videoAudio, self.videoSentence, self.trainVid, \
                self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
                self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
                self.testVid, self.aaa = pickle.load(
                    open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        self.feature_types = feature_types

    def __getitem__(self, index):
        vid = self.keys[index]
        if self.feature_types == FeatureType.TEXT:
            tensors = [torch.FloatTensor(self.videoText[vid])]
        elif self.feature_types == FeatureType.AUDIO:
            tensors = [torch.FloatTensor(self.videoAudio[vid])]
        elif self.feature_types == FeatureType.VISUAL:
            tensors = [torch.FloatTensor(self.videoVisual[vid])]
        elif self.feature_types == FeatureType.TEXT_VISUAL:
            tensors = [torch.FloatTensor(
                self.videoText[vid]), torch.FloatTensor(self.videoVisual[vid])]
        elif self.feature_types == FeatureType.TEXT_AUDIO:
            tensors = [torch.FloatTensor(
                self.videoText[vid]), torch.FloatTensor(self.videoAudio[vid])]
        elif self.feature_types == FeatureType.AUDIO_VISUAL:
            tensors = [torch.FloatTensor(
                self.videoAudio[vid]), torch.FloatTensor(self.videoVisual[vid])]
        elif self.feature_types == FeatureType.TEXT_AUDIO_VISUAL:
            tensors = [torch.FloatTensor(self.videoText[vid]), torch.FloatTensor(
                self.videoAudio[vid]), torch.FloatTensor(self.videoVisual[vid])]
        else:
            raise ValueError('Invalid FeatureType')
        tensors.extend([
            torch.FloatTensor(self.videoSpeakers[vid]),
            torch.FloatTensor([1]*len(self.videoLabels[vid])),
            torch.LongTensor(self.videoLabels[vid]),
            vid
        ])

        # convert to tuple to make items are unpacked as separate outputs
        return tuple(tensors)
        # return torch.FloatTensor(self.videoText[vid]), \
        #     torch.FloatTensor(self.videoVisual[vid]), \
        #     torch.FloatTensor(self.videoAudio[vid]), \
        #     torch.FloatTensor(self.videoSpeakers[vid]), \
        #     torch.FloatTensor([1]*len(self.videoLabels[vid])), \
        #     torch.LongTensor(self.videoLabels[vid]), \
        #     vid

    def __len__(self):
        return self.len

    def collate_fn(self, data) -> list[torch.Tensor | list]:
        """
        Collates a batch of data for the DataLoader.
        Args:
            data (list): A list of data samples, where each sample is a list or tuple of features.
        Returns:
            list: A list where each element is a padded sequence if the index is less than the number of feature types plus 3,
                  otherwise, it is a list of the original data values.
        """

        dat = pd.DataFrame(data)
        n_types = FeatureType.get_numof_types(self.feature_types)
        lower_bound = n_types + 1
        upper_bound = n_types + 3
        return [pad_sequence(dat[i]) if i < lower_bound else pad_sequence(dat[i], True) if i < upper_bound else dat[i].tolist() for i in dat]
