import os
from turtle import width
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# from data.repr_utils.event_voxel import EventVoxel
# from data.repr_utils.hats import Hats
# from data.repr_utils.hots import HOTS
from data.repr_utils.time_surface import TimeSurface

class Representation(Dataset):
    def __init__(self,
                 representation_type: str,
                 dataset_dir: str,
                 height: int,
                 width: int,
                 use_polarity: bool,
                 events_downsample_ratio: int,
                 purpose: str,
                 tau: Optional[float] = None,
                 normalize: str = 'None',
                 cache_root: Optional[str] = None,
                 use_cache: bool = True):
        assert representation_type.lower() in ['event_voxel', 'time_surface', 'hots', 'hats'], \
            f"Unsupported representation type: {representation_type}"
        assert purpose.lower() in ['train', 'val', 'test'], \
            f"Unsupported purpose: {purpose}"
        
        self.representation_type = representation_type.lower()

        if self.representation_type == 'event_voxel':
            self.representation = EventVoxel(dataset_dir, purpose)
        elif self.representation_type == 'time_surface':
            self.representation = TimeSurface(dataset_dir,
                                              height, 
                                              width,
                                              use_polarity,
                                              use_cache,
                                              cache_root,
                                              purpose,
                                              events_downsample_ratio,
                                              tau,
                                              normalize)
        elif self.representation_type == 'hots':
            self.representation = HOTS(dataset_dir, purpose)
        elif self.representation_type == 'hats':
            self.representation = Hats(dataset_dir, purpose)
        
        

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, idx):
        #TODO: Apply different encoder
        return {'representation': self.representation_type,
                'data': self.representation[idx]['data'],
                'label': self.representation[idx]['label'],
                'path': self.representation[idx]['path']}
    

if __name__ == "__main__":
    dataset_dir = '/fs/nexus-scratch/tuxunlu/git/EventRepContrastiveLearning/N_Imagenet'
    dataset = Representation(representation_type='time_surface',
                            dataset_dir=dataset_dir,
                            width=680, height=680,
                            tau=0.5,
                            events_downsample_ratio=2,
                            use_cache=True,
                            cache_root='/fs/nexus-scratch/tuxunlu/git/EventRepContrastiveLearning/cache',
                            use_polarity=True,
                            purpose='train')
    print(f"Dataset size: {len(dataset)}")
    print("Data shape:", dataset[0]['data'].shape)
    print("Label:", dataset[0]['label'])
    print("Path:", dataset[0]['path'])



