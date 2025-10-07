import os
from numba import njit, prange
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.n_imagenet import NImageNet as Preprocessor
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
class HATS(Dataset):
    def __init__(self, dataset_dir,
                 width: int,
                 height: int,
                 tau: float,
                 R: int,
                 K: int,
                 cache_root: str,
                 purpose='train',
                 use_cache=True,
                 normalize='None',
                 use_polarity=True):
        assert purpose in ['train', 'test'], "Split must be either 'train' or 'test'."

        self.dataset_dir = dataset_dir
        self.preprocessor = Preprocessor(dataset_dir=dataset_dir, split=purpose)
        self.purpose = purpose
        self.use_cache = use_cache
        self.cache_root = cache_root
        self.width = width
        self.height = height
        self.target_W = K
        self.target_H = K
        self.tau = tau
        self.R = R
        self.K = K
        self.normalize = normalize
        
        self.num_cell_width = (width // K)   # number of cells horizontally
        self.num_cell_height = (height // K) # number of cells vertically
        self.n_cells = self.num_cell_width * self.num_cell_height
        self.n_polarities = 2

        self.cell_per_pixel_last_timestamps = np.zeros((self.n_cells, self.n_polarities, 2*self.R+1, 2*self.R+1), dtype=np.float32)
        self.cell_per_pixel_sum = np.zeros((self.n_cells, self.n_polarities, 2*self.R+1, 2*self.R+1), dtype=np.float32)
        self.event_count = np.zeros((self.n_cells, self.n_polarities), dtype=np.int32)

    @staticmethod
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        valid_len = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
        padded_sequences = pad_sequence(sequences, batch_first=True)
        return padded_sequences, valid_len, torch.stack(labels)

    def __len__(self):
        return len(self.preprocessor)
    
    def __getitem__(self, idx):
        item_dict = self.preprocessor[idx]
        events_t, events_xy, events_p, label, path = item_dict['events_t'], item_dict['events_xy'], item_dict['events_p'], item_dict['label'], item_dict['path']

        # Build cache path
        rel_seq_path = os.path.relpath(path, start=self.dataset_dir)
        cache_dir_path = os.path.join(self.cache_root, rel_seq_path, 'HATS')
        cache_name = f"HAT_tau={self.tau}_R={self.R}_K={self.K}.pt"
        cached_path = os.path.join(cache_dir_path, cache_name)

        if self.use_cache and os.path.exists(cached_path):
            return_dict = torch.load(cached_path, weights_only=False)
            # print(f"Loaded cached Time Surface from: {cached_path}")
            return return_dict
        
        hats = []
        
        t_out, xy_out, p_out, offsets = split_events_KxK_numba(
            events_t, events_xy, events_p,
            self.height, self.width, self.num_cell_width
        )

        output = []
        hats_poss = []
        hats_negs = []

        for cell_id in tqdm(range(self.n_cells), desc=f"Processing HATS for sample {idx}"):
            start_idx = offsets[cell_id]
            end_idx = offsets[cell_id + 1]
            if start_idx >= end_idx:
                continue
            events_t_cell = t_out[start_idx:end_idx]
            events_xy_cell = xy_out[start_idx:end_idx]
            events_p_cell = p_out[start_idx:end_idx]

            # compute HATS for this cell
            hats_pos, hats_neg = self.process(events_t_cell, events_xy_cell, events_p_cell)
            # output.append(out)
            hats_poss.append(hats_pos)
            hats_negs.append(hats_neg)
        
        hats_poss = np.stack(hats_poss, axis=0)  # (n_cells, H, W)
        hats_negs = np.stack(hats_negs, axis=0)  # (n_cells, H, W)
        hats = np.stack([hats_poss, hats_negs], axis=1)  # (n_cells, 2, H, W)
        hats = normalize_array(hats, self.normalize)

        # output = np.stack(output, axis=0)  # (n_cells, 2, H, W)
        # C, P, H, W = output.shape
        # output = output.reshape(C*P, H, W)  # (n_cells*2, H, W)
        # hats.append(torch.from_numpy(output).float())

        # hats_poss = np.stack(hats_poss, axis=0)  # (n_cells, H, W)
        # fig, axes = plt.subplots(self.num_cell_width, self.num_cell_height, figsize=(12, 12))
        # axes = axes.flatten()
        # print(hats_poss.shape)
        # for j in range(self.num_cell_width * self.num_cell_height):
        #     axes[j].imshow(hats_poss[j], cmap='hot')
        #     axes[j].axis("off")
        
        # plt.subplots_adjust(wspace=0.05, hspace=0.05)
        # plt.savefig(f"./{class_name}_{counter}_hats.png", dpi=300, bbox_inches='tight')
        # plt.close(fig)

        return_dict = {
            'data': hats,
            'label': label,
            'path': path
        }

        if self.use_cache:
            os.makedirs(cache_dir_path, exist_ok=True)
            torch.save(return_dict, cached_path)
            print(f"Saved HATS to cache: {cached_path}")

        return return_dict
    
    def process(self, events_t, events_xy, events_p):
        """
        events: [N,4] (x,y,t,p)
        """
        
        pos_idx = np.where(events_p == 1)[0]
        neg_idx = np.where(events_p == 0)[0]

        hats_pos, _ = create_hats(
            source_H=self.height,
            source_W=self.width,
            target_H=self.target_H,
            target_W=self.target_W,
            events=(events_t[pos_idx], events_xy[pos_idx]),
            tau=self.tau,
            t_query=None,
            output_index=False,
            normalize='None',  # normalize after stacking if desired
        )
        hats_neg, _ = create_hats(
            source_H=self.height,
            source_W=self.width,
            target_H=self.target_H,
            target_W=self.target_W,
            events=(events_t[neg_idx], events_xy[neg_idx]),
            tau=self.tau,
            t_query=None,
            output_index=False,
            normalize='None',
        )

        hats_pos = normalize_array(hats_pos, self.normalize)
        hats_neg = normalize_array(hats_neg, self.normalize)

        # hats_stack = np.stack([hats_pos, hats_neg], axis=0)  # (2, H, W)
        # hats_stack = normalize_array(hats_stack, self.normalize)

        return hats_pos, hats_neg
    
import numpy as np
from numba import njit

@njit(parallel=True, cache=True, fastmath=True)
def split_events_KxK_numba(t, xy, p, H, W, K):
    """
    Split events into KxK grid cells. Output arrays are grouped contiguously
    by cell (row-major), and xy_out contains *cell-local* coordinates where
    each cell's top-left corner is (0, 0).

    Returns:
        t_out  : (N,)
        xy_out : (N, 2)  with local (x, y) per cell
        p_out  : (N,)
        offsets: (K*K + 1,) prefix-sum; events for cell c are in [offsets[c], offsets[c+1])
    """
    N = t.shape[0]
    counts = np.zeros(K * K, np.int64)

    # cell sizes using ceil division so we cover the whole image
    cell_w = (W + K - 1) // K
    cell_h = (H + K - 1) // K

    # ---- pass 1: histogram counts per cell ----
    for i in range(N):
        x = xy[i, 0]
        y = xy[i, 1]
        cx = x // cell_w
        cy = y // cell_h
        if cx >= K: cx = K - 1
        if cy >= K: cy = K - 1
        cid = cy * K + cx
        counts[cid] += 1

    # ---- prefix sums => offsets ----
    offsets = np.empty(K * K + 1, np.int64)
    offsets[0] = 0
    for k in range(K * K):
        offsets[k + 1] = offsets[k] + counts[k]

    # ---- pass 2: scatter with cell-local coordinates ----
    write_ptr = offsets[:-1].copy()
    t_out  = np.empty(N, t.dtype)
    xy_out = np.empty((N, 2), xy.dtype)
    p_out  = np.empty(N, p.dtype)

    for i in range(N):
        x = xy[i, 0]
        y = xy[i, 1]

        # which cell?
        cx = x // cell_w
        cy = y // cell_h
        if cx >= K: cx = K - 1
        if cy >= K: cy = K - 1
        cid = cy * K + cx

        # cell top-left in global coords
        cell_x0 = cx * cell_w
        cell_y0 = cy * cell_h

        # cell size (last row/col may be smaller)
        max_w = W - cell_x0
        if max_w > cell_w:
            max_w = cell_w
        max_h = H - cell_y0
        if max_h > cell_h:
            max_h = cell_h

        # convert to cell-local coords and clamp to cell bounds
        x_local = x - cell_x0
        y_local = y - cell_y0
        if x_local < 0: x_local = 0
        if y_local < 0: y_local = 0
        if x_local >= max_w: x_local = max_w - 1
        if y_local >= max_h: y_local = max_h - 1

        j = write_ptr[cid]
        write_ptr[cid] = j + 1

        t_out[j] = t[i]
        xy_out[j, 0] = x_local
        xy_out[j, 1] = y_local
        p_out[j] = p[i]

    return t_out, xy_out, p_out, offsets

def create_hats(source_H: int,
                source_W: int,
                target_H: int,
                target_W: int,
                events: tuple[np.ndarray, np.ndarray],
                tau: float,
                t_query: float | None = None,
                output_index: bool = False,
                normalize: str = 'None',
                avg_radius: int = 1):
    """
    HATS for ONE cell/neighborhood (events already localized to this patch).

    Returns:
        hats: (target_H, target_W) float32 in [0,1] after averaging & optional normalization.
        last_idx_map (optional): (target_H, target_W) int64 of last-event indices *before* averaging.
    """
    t, xy = events

    # Empty-cell fast path
    if t.size == 0:
        hats = np.zeros((target_H, target_W), dtype=np.float32)
        if output_index:
            return hats, -np.ones((target_H, target_W), dtype=np.int64)
        return hats, None

    # Events are already local to the neighborhood; just clamp
    x = np.clip(xy[:, 0].astype(np.int64), 0, target_W - 1)
    y = np.clip(xy[:, 1].astype(np.int64), 0, target_H - 1)

    if t_query is None:
        t_query = float(t[-1])

    # Last event time per pixel inside this cell
    last_time_flat, last_idx_flat = _last_event_time_per_pixel(
        target_H, target_W, x, y, t.astype(np.float64)
    )

    # Time surface (delta in microseconds -> decay by tau)
    delta = (t_query - last_time_flat) / 1.0e6 / float(tau)
    ts = np.exp(-delta).reshape(target_H, target_W).astype(np.float32)

    # Optional normalization (usually unnecessary for TS/HATS)
    ts = normalize_array(ts, normalize)

    if output_index:
        return ts, last_idx_flat.reshape(target_H, target_W)
    return ts, None


def _last_event_time_per_pixel(target_H: int,
                               target_W: int,
                               x_ds: np.ndarray,
                               y_ds: np.ndarray,
                               t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized computation of last event *time* and *index* per pixel.

    Returns:
        last_time_map_flat: (target_H*target_W,) float64 of last timestamps; pixels with no events set to a very negative value.
        last_idx_map_flat:  (target_H*target_W,) int64 of last event indices in the original arrays; -1 for no events.
    """
    total = target_H * target_W
    if t.size == 0:
        # no events in chunk
        last_time_map_flat = np.full(total, -1e20, dtype=np.float64)
        last_idx_map_flat = np.full(total, -1, dtype=np.int64)
        return last_time_map_flat, last_idx_map_flat

    # Flattened pixel indices in downsampled grid
    pix_idx = (y_ds.astype(np.int64) * target_W + x_ds.astype(np.int64))

    # Stable lexsort by (pix_idx, time)
    order = np.lexsort((t, pix_idx))
    pix_sorted = pix_idx[order]
    t_sorted = t[order]

    # Find ends of each group (last occurrence is the last event at that pixel)
    if pix_sorted.size == 0:
        last_time_map_flat = np.full(total, -1e20, dtype=np.float64)
        last_idx_map_flat = np.full(total, -1, dtype=np.int64)
        return last_time_map_flat, last_idx_map_flat

    boundaries = np.flatnonzero(np.diff(pix_sorted))
    group_end_pos = np.concatenate([
        boundaries, np.array([pix_sorted.size - 1], dtype=np.int64)
    ])

    last_pix = pix_sorted[group_end_pos]
    last_time = t_sorted[group_end_pos]
    last_orig_idx = order[group_end_pos]

    # Fill maps
    last_time_map_flat = np.full(total, -1e20, dtype=np.float64)
    last_idx_map_flat = np.full(total, -1, dtype=np.int64)
    last_time_map_flat[last_pix] = last_time
    last_idx_map_flat[last_pix] = last_orig_idx

    return last_time_map_flat, last_idx_map_flat


def normalize_array(arr: np.ndarray, normalize: str):
    """
    Normalize a numpy array in-place.
    Args:
        arr: np.ndarray
        normalize: 'standardization' | 'normalization' | 'None'
    Returns:
        np.ndarray
    """
    arr = arr.astype(np.float32)
    eps = 1e-8
    if normalize == 'standardization':
        mean = arr.mean()
        std = arr.std()
        arr = (arr - mean) / (std + eps)
    elif normalize == 'normalization':
        min_val = arr.min()
        max_val = arr.max()
        arr = (arr - min_val) / (max_val - min_val + eps)
    elif normalize == 'None':
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")
    return arr



def main():
    import time
    dataset_dir = '/fs/nexus-projects/DVS_Actions/NatureRoboticsData/'
    dataset = HATS(dataset_dir,
                   width=680, height=680,
                   tau=0.5,
                   R=170,
                   K=170,
                   cache_root="/fs/nexus-projects/DVS_Actions/NatureRoboticsDataCache",
                   use_cache=True,
                   purpose="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    dataloader_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        start_time = time.perf_counter()
        padded_sequences, labels = next(dataloader_iter)
        proc_time = time.perf_counter() - start_time
        print(f"Class: {labels}, Frames: {padded_sequences.shape}")
        print(f"Process time: {proc_time:.4f} s")
        break


if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
