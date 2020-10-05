"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # Create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. Verify if method is 
        # correct by running it on one of the volumes in training set and comparing 
        # with the label in 3D Slicer.
        slices = np.zeros(volume.shape)

        for slice_idx in range(volume.shape[0]):
            # get slice, normalize, and create tensor
            slice_2d = volume[slice_idx,:,:] 
            slice_2d = slice_2d.astype(np.single)/255.0
            slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)
            #print('slice_tensor: ', slice_tensor.shape)
            # get prediction
            prediction = self.model(slice_tensor.to(self.device))  # 1 x 3 x 64 x 64
            pred = np.squeeze(prediction.cpu().detach())   # 3 x 64 x 64
            #print(f'Inference, prediction: {prediction.shape}, pred: {pred.shape}')
            slices[slice_idx,:,:] = torch.argmax(pred, dim=0).numpy()  # each pixel has one pred (0,1,2)
            
        return slices
