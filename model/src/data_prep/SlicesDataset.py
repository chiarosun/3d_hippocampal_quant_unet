"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]  # this is a batch (8 images) (image batch_item_number, slice_in_image)
        sample = dict()
        sample["id"] = idx

        # Could implement caching strategy here if dataset is too large to fit
        # in memory entirely
        # Also this would be the place to call transforms if data augmentation is used
        
        # Create two new keys in the "sample" dictionary, named "image" and "seg"
        # The values are 3D Torch Tensors with image and label data respectively. 
        # First dimension is size 1, and last two hold the voxel data from the respective
        # slices. Stor the 2D slice data in the last 2 dimensions of the 3D Tensors. 
        # Tensor needs to be of shape [1, patch_size, patch_size]

        # get each slice per vol in sagittal plane (will be a 64x64 patch)
        # then unsqueeze to get the first dimension

        sample['image'] = torch.from_numpy(self.data[slc[0]]['image'][slc[1],:,:]).unsqueeze(0).cuda()
        sample['seg'] = torch.from_numpy(self.data[slc[0]]['seg'][slc[1],:,:]).unsqueeze(0).cuda()

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
