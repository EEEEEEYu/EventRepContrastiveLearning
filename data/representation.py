from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from data.repr_utils.event_voxel import EventVoxel
# from data.repr_utils.hats import Hats
from data.repr_utils.hots import HOTS
from data.repr_utils.time_surface import TimeSurface

class Representation(Dataset):
    def __init__(self,
                 representation_type: str,
                 dataset_dir: str,
                 height: int,
                 width: int,
                 use_polarity: bool,
                 purpose: str,
                 cache_root: Optional[str] = None,
                 use_cache: bool = True,
                 **kwargs):
        assert representation_type.lower() in ['event_voxel', 'time_surface', 'hots', 'hats'], \
            f"Unsupported representation type: {representation_type}"
        assert purpose.lower() in ['train', 'val', 'test'], \
            f"Unsupported purpose: {purpose}"
        
        self.representation_type = representation_type.lower()

        if self.representation_type == 'event_voxel':
            self.representation = EventVoxel(dataset_dir,
                                       purpose=purpose,
                                       height=height,
                                       width=width,
                                       use_polarity=use_polarity,
                                       use_cache=use_cache,
                                       cache_root=cache_root,
                                       **kwargs)
        elif self.representation_type == 'time_surface':
            self.representation = TimeSurface(dataset_dir,
                                                purpose=purpose,
                                                height=height,
                                                width=width,
                                                use_polarity=use_polarity,
                                                use_cache=use_cache,
                                                cache_root=cache_root,
                                                **kwargs)
        elif self.representation_type == 'hots':
            self.representation = HOTS(dataset_dir,
                                       purpose=purpose,
                                       height=height,
                                       width=width,
                                       use_polarity=use_polarity,
                                       use_cache=use_cache,
                                       cache_root=cache_root,
                                       **kwargs)
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

    # dataset_ts = Representation(representation_type='time_surface',
    #                         dataset_dir=dataset_dir,
    #                         width=680, height=680,
    #                         tau=0.5,
    #                         events_downsample_ratio=2,
    #                         use_cache=True,
    #                         cache_root='/fs/nexus-scratch/tuxunlu/git/EventRepContrastiveLearning/cache',
    #                         use_polarity=True,
    #                         purpose='train')
    # print(f"Dataset size: {len(dataset_ts)}")
    # print("Data shape:", dataset_ts[0]['data'].shape)
    # print("Label:", dataset_ts[0]['label'])
    # print("Path:", dataset_ts[0]['path'])

    # dataset_hots = Representation(representation_type='hots',
    #                             dataset_dir=dataset_dir,
    #                             width=680, height=680,
    #                             events_downsample_ratio=2,
    #                             use_cache=True,
    #                             cache_root='/fs/nexus-scratch/tuxunlu/git/EventRepContrastiveLearning/cache',
    #                             use_polarity=True,
    #                             purpose='train',
    #                             N_1 = 4, tau_1 = 20000.0, r_1 = 2,
    #                             K_N = 2, K_tau = 2.0, K_r = 2,
    #                             prototype_cache_root = "./cache/hots_prototypes",
    #                             max_slices_for_proto = 2,
    #                         )
    # print(f"Dataset size: {len(dataset_hots)}")
    # print("Data shape:", dataset_hots[0]['data'].shape)
    # print("Label:", dataset_hots[0]['label'])
    # print("Path:", dataset_hots[0]['path']) 

    dataset_ev = Representation(representation_type='event_voxel',
                                dataset_dir=dataset_dir,
                                width=680, height=680,
                                num_bins=9,
                                accumulation_interval_ms=50.0,
                                events_downsample_ratio=2,
                                use_cache=True,
                                cache_root='/fs/nexus-scratch/tuxunlu/git/EventRepContrastiveLearning/cache',
                                use_polarity=True,
                                purpose='train',
                            )
    print(f"Dataset size: {len(dataset_ev)}")
    print("Data shape:", dataset_ev[0]['data'].shape)
    print("Label:", dataset_ev[0]['label'])
    print("Path:", dataset_ev[0]['path'])





