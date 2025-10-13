import inspect
from typing import Dict, List, Optional
from torch.utils.data import Dataset

from data.repr_utils.event_voxel import EventVoxel
from data.repr_utils.hats import HATS
from data.repr_utils.hots import HOTS
from data.repr_utils.time_surface import TimeSurface

REPRESENTATION_CLASSES = {
    "event_voxel": EventVoxel,
    "time_surface": TimeSurface,
    "hots": HOTS,
    "hats": HATS,
}

class Representation(Dataset):
    def __init__(self,
                 representation_names: List[str],
                 dataset_dir: str,
                 height: int,
                 width: int,
                 purpose: str,
                 cache_root: Optional[str] = None,
                 use_cache: bool = True,
                 # --- EventVoxel-specific ---
                 event_voxel_num_bins: int = 10,
                 event_voxel_use_polarity: bool = True,
                 event_voxel_spatial_downsample_ratio: float = 1.0,
                 # --- TimeSurface-specific ---
                 time_surface_use_polarity: bool = True,
                 time_surface_spatial_downsample_ratio: float = 1.0,
                 time_surface_tau: float = 0.5,
                 time_surface_normalize: str = 'None',
                 # --- HATS/HOTS/etc. can add more later ---
                 HATS_tau: float = 0.5,
                 HATS_R: float = 85,
                 HATS_K: float = 85,
                 HOTS_tau_1: float = 20000.0,
                 HOTS_N_1: int = 4,
                 HOTS_r_1: int = 2,
                 HOTS_K_N: int = 2,
                 HOTS_K_tau: float = 2.0,
                 HOTS_K_r: int = 2
                 ):
        self.representation_instances = []
        self.representation_names = representation_names

        # get all args passed to this init
        init_args = locals().copy()
        init_args.pop("self")

        for name in representation_names:
            name_lower = name.lower()
            assert name_lower in REPRESENTATION_CLASSES, f"Unsupported representation: {name}"
            assert purpose.lower() in ['train', 'validation', 'test'], f"Unsupported purpose: {purpose}"

            cls = REPRESENTATION_CLASSES[name_lower]

            # Extract parameters specific to this representation
            prefix = f"{name_lower}_"
            class_kwargs = {
                k: v
                for k, v in init_args.items()
                if k.startswith(prefix)
            }

            # Filter by valid args of the class constructor
            sig = inspect.signature(cls.__init__)
            valid_params = sig.parameters.keys()
            class_kwargs = {k: v for k, v in class_kwargs.items() if k in valid_params}

            # Instantiate
            instance = cls(
                dataset_dir=dataset_dir,
                height=height,
                width=width,
                purpose=purpose,
                use_cache=use_cache,
                cache_root=cache_root,
                **class_kwargs,
            )
            self.representation_instances.append(instance)

    def __len__(self):
        return len(self.representation_instances[0])

    def __getitem__(self, idx):
        repr_data = []
        label = None
        path = None

        for representation in self.representation_instances:
            result = representation[idx]
            repr_data.append(result['data'])
            label = result['label']
            path = result['path']

        return {
            'representation': self.representation_names,
            'repr_data': repr_data,
            'label': label,
            'path': path,
        }
