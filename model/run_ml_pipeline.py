"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from sklearn.model_selection import train_test_split

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r""
        #self.n_epochs = 10
        self.n_epochs = 1
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = ""

if __name__ == "__main__":
    # Get configuration

    # Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()
    #c.root_dir = '/home/magellan/Projects/uda_ai_healthcare/Unit2-3D-Imaging/Final_Project/gh_chiaro_stage/src/section1/out'
    c.root_dir = './../../eda/out'
    c.test_results_dir = r"./test_results"

    # Load data
    print("Loading data...")

    # LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))
    print('keys: ', type(keys), keys)
    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # Create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    split['train'], val_test = train_test_split(keys, test_size=0.3, random_state=100)
    split['val'], split['test'] = train_test_split(val_test, test_size=0.5, random_state=100)
    print(len(split['train']), len(split['test']), len(split['val']))
    print(len(split['train'])/260, len(split['test'])/260, len(split['val'])/260)


    # Set up and run experiment
    exp = UNetExperiment(c, split, data)

    # could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing
    results_json = exp.run_test()
    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

