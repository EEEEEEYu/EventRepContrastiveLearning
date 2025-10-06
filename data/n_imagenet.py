import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NImageNet(Dataset):
    def __init__(self, dataset_dir, split):
        assert os.path.exists(dataset_dir), f"Error: Dataset folder does not exist! Input folder: {dataset_dir}"
        assert split in ('train', 'validation', 'test'), f"Error: Unknown split method: {split}"
        self.dataset_dir = dataset_dir
        self.split = split

        # Part_1 ... Part_10
        parts_prefixes = [
            os.path.join(self.dataset_dir, split, f"Part_{i}") for i in range(1, 11)
            if os.path.exists(os.path.join(self.dataset_dir, split, f"Part_{i}"))
        ]

        # Each class folder under each Part
        class_folder_prefixes = []
        for part_folder in parts_prefixes:
            for class_folder in os.listdir(part_folder):
                class_folder_path = os.path.join(part_folder, class_folder)
                if os.path.isdir(class_folder_path):
                    class_folder_prefixes.append(class_folder_path)

        # List of (class_folder_path, [instance_files])
        self.all_instance_list = [
            (class_folder, [
                f for f in os.listdir(class_folder) if f.endswith(".npz")
            ]) for class_folder in class_folder_prefixes
        ]

        # Map class id string -> integer id
        mapped_id = 0
        self.class_id_dict = dict()
        for class_id_path in class_folder_prefixes:
            class_id = os.path.basename(class_id_path)
            if class_id not in self.class_id_dict:
                self.class_id_dict[class_id] = mapped_id
                mapped_id += 1

        # Prefix count list to index samples globally
        self.prefix_count = []
        total = 0
        for _, instance_list in self.all_instance_list:
            total += len(instance_list)
            self.prefix_count.append(total)

    def get_path_from_global_idx(self, idx):
        # Find which class folder contains this global index
        instance_list_idx = np.searchsorted(self.prefix_count, idx + 1)
        prev_prefix = 0 if instance_list_idx == 0 else self.prefix_count[instance_list_idx - 1]
        local_idx = idx - prev_prefix

        instance_prefix, instance_files = self.all_instance_list[instance_list_idx]
        instance_file = instance_files[local_idx]
        return os.path.join(instance_prefix, instance_file)

    @staticmethod
    def load_npz_into_events(path):
        """
        Load N-ImageNet .npz file and return three numpy arrays:
        events_xy: (N, 2), dtype=np.uint16
        events_t:  (N,),   dtype=np.uint16
        events_p:  (N,),   dtype=np.uint8
        """
        with np.load(path, allow_pickle=True) as data:
            # The key might be 'event_data'
            if 'event_data' not in data:
                raise KeyError(f"No 'event_data' key found in {path}")
            arr = data['event_data']

            # Ensure structured dtype with x,y,t,p
            x = arr['x'].astype(np.uint16)
            y = arr['y'].astype(np.uint16)
            t = arr['t'].astype(np.uint16)
            p = arr['p'].astype(np.uint8)

            events_xy = np.stack([x, y], axis=-1)
            events_t = t
            events_p = p
            return events_t, events_xy, events_p

    def __len__(self):
        return self.prefix_count[-1] if len(self.prefix_count) > 0 else 0

    def __getitem__(self, idx):
        npz_path = self.get_path_from_global_idx(idx)
        events_t, events_xy, events_p = self.load_npz_into_events(npz_path)
        class_id = os.path.basename(os.path.dirname(npz_path))
        label = self.class_id_dict[class_id]

        return_dict = {
            'events_t': events_t,
            'events_xy': events_xy,
            'events_p': events_p,
            'label': label,
            'path': npz_path
        }

        return return_dict


if __name__ == "__main__":
    dataset = NImageNet("/fs/nexus-scratch/tuxunlu/git/EventRepContrastiveLearning/N_Imagenet", split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for batch in loader:
        events_t = batch['events_t']
        events_xy = batch['events_xy']
        events_p = batch['events_p']
        class_id = batch['label']
        path = batch['path']
        print(f"t shape: {events_t[0].shape}, xy shape: {events_xy[0].shape}, p shape: {events_p[0].shape}, class_id: {class_id}", f"path: {path}")
        break
