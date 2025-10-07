# hots_dataset.py
import os
import math
import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pathlib
from PIL import Image
import cv2
_HAS_CV2 = True

# --- HOTS repo functions (use exactly these names) ---
from data.repr_utils.hots_utils.noise_filter import remove_isolated_pixels
from data.repr_utils.hots_utils.layer_operations import (
    initialise_time_surface_prototypes,
    train_layer,
    generate_layer_outputs,
)

from data.n_imagenet import NImageNet as Preprocessor

def normalize_array(arr: np.ndarray, normalize: str):
    arr = arr.astype(np.float32)
    eps = 1e-8
    if normalize == 'standardization':
        return (arr - arr.mean()) / (arr.std() + eps)
    if normalize == 'normalization':
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + eps)
    if normalize == 'None':
        return arr
    raise ValueError(f"Unknown normalization method: {normalize}")


# Minimal event object that matches what HOTS lib expects (fields: x, y, ts, p)
class _Evt:
    __slots__ = ("x", "y", "ts", "p")
    def __init__(self, x, y, ts, p):
        self.x = int(x); self.y = int(y); self.ts = int(ts); self.p = int(p)


def _arrays_to_event_list(t: np.ndarray, xy: np.ndarray, p: np.ndarray) -> list[_Evt]:
    # Ensure lengths align
    n = len(t)
    if xy.shape[0] != n or len(p) != n:
        raise ValueError("t, xy, p must have the same length.")
    # Build list of event objects (order preserved)
    return [_Evt(xy[i, 0], xy[i, 1], t[i], p[i]) for i in range(n)]


def _histogram_over_prototypes(events: list[_Evt], n_bins: int) -> np.ndarray:
    """Single fixed-length feature per slice: counts over L3 prototype index (stored in .p)."""
    if not events:
        return np.zeros((n_bins,), dtype=np.float32)
    idx = np.fromiter((e.p for e in events), count=len(events), dtype=np.int64)
    return np.bincount(idx, minlength=n_bins).astype(np.float32)

def _events_to_arrays(events: list[_Evt]):
    if not events:
        return (np.empty((0,), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
    t = np.fromiter((e.ts for e in events), count=len(events), dtype=np.int64)
    xy = np.empty((len(events), 2), dtype=np.int64)
    p  = np.fromiter((e.p  for e in events), count=len(events), dtype=np.int64)
    for i, e in enumerate(events):
        xy[i, 0] = e.x; xy[i, 1] = e.y
    return t, xy, p

def _l3_events_to_labelmap(events: list[_Evt], H: int, W: int, n_bins: int, mode: str = "last"):
    """
    Returns (H, W) int map of prototype index per pixel.
    mode='last'  : last event wins at each pixel
    mode='vote'  : argmax over counts per prototype at each pixel
    """
    if mode == "last":
        lab = np.full((H, W), fill_value=-1, dtype=np.int16)
        for e in events:
            if 0 <= e.y < H and 0 <= e.x < W:
                lab[e.y, e.x] = e.p
        return lab
    elif mode == "vote":
        counts = np.zeros((H, W, n_bins), dtype=np.int32)
        for e in events:
            if 0 <= e.y < H and 0 <= e.x < W and 0 <= e.p < n_bins:
                counts[e.y, e.x, e.p] += 1
        lab = counts.argmax(axis=2).astype(np.int16)
        # keep pixels with no events as -1
        empty = (counts.sum(axis=2) == 0)
        lab[empty] = -1
        return lab
    else:
        raise ValueError("mode must be 'last' or 'vote'")

def _save_labelmap_png(labelmap: np.ndarray, out_png: str, n_bins: int):
    """
    Saves a colored PNG for the integer labelmap (−1 => black).
    Uses cv2 if available; otherwise a simple PIL palette.
    """
    lm = labelmap.copy()
    lm = np.where(lm < 0, n_bins, lm)  # map -1 to "background" color index n_bins
    if _HAS_CV2:
        # scale to 0..255 and colorize; background set to 0 then overwritten to black
        norm = (lm.astype(np.float32) / max(n_bins, 1)) * 255.0
        norm = norm.astype(np.uint8)
        img = cv2.applyColorMap(norm, cv2.COLORMAP_HSV)
        img[lm == n_bins] = (0, 0, 0)  # black for background
        cv2.imwrite(out_png, img[:, :, ::-1])  # BGR->RGB for consistency
    else:
        # indexed palette image in PIL
        pal_img = Image.fromarray(lm.astype(np.uint8), mode="P")
        # build a simple HSV-like palette
        palette = []
        for k in range(n_bins):
            hue = int(255 * k / max(1, n_bins))
            palette += [hue, 255, 255]  # will look funky in "P", but OK
        palette += [0, 0, 0]  # background at index n_bins
        # fill up to 256*3 entries
        palette += [0, 0, 0] * (256 - len(palette)//3)
        pal_img.putpalette(palette, rawmode="HSV")
        pal_img.save(out_png)


class HOTS(data.Dataset):
    """
    HOTS dataset with the same structure as your TimeSurface class:
      - Uses Preprocessor to slice sequences into temporal chunks
      - Per-sequence caching under cache_root/<rel_seq_path>/hots/
      - Returns one feature vector per chunk (so a sequence is [L, D])

    Output per sample:
      features: torch.FloatTensor of shape (L, D) where D = N_3 (number of layer-3 prototypes)
      label:    torch.LongTensor scalar
    """

    def __init__(
        self,
        dataset_dir: str,
        height: int,
        width: int,
        use_polarity: bool,
        use_cache: bool,
        cache_root: str,
        purpose: str,
        normalize: str = 'None',

        # HOTS-specific hyperparameters (exact symbols as the repo uses)
        N_1: int = 4, tau_1: float = 20000.0, r_1: int = 2,
        K_N: int = 2, K_tau: float = 2.0, K_r: int = 2,

        # Noise filter params (same function as the repo calls)
        eps: int = 12, min_samples: int = 20,

        # Prototype training control
        max_slices_for_proto: int = 100,  # cap the number of slices used to learn prototypes
        prototype_cache_root: str | None = None,

        debug_max_slices: int | None = None,  # max slices per sequence to save debug files for
        debug_save_l3: bool = False,
        debug_root: str | None = None,
        debug_mode: str = "last",    # 'last' or 'vote'

        denoise: bool = False,
    ):
        # --- mirror your TimeSurface init ---
        self.dataset_dir = dataset_dir
        self.preprocessor = Preprocessor(dataset_dir=dataset_dir, split=purpose)
        self.height = height
        self.width = width
        self.use_polarity = use_polarity  # if False, all events treated as a single polarity (p=1)
        self.use_cache = use_cache
        self.cache_root = cache_root
        self.purpose = purpose
        self.normalize = normalize
        self.denoise = denoise

        # HOTS params (exactly like the repo main pipeline)
        self.N_1, self.tau_1, self.r_1 = int(N_1), float(tau_1), int(r_1)
        self.K_N, self.K_tau, self.K_r = int(K_N), float(K_tau), int(K_r)

        self.N_2 = self.N_1 * self.K_N
        self.tau_2 = self.tau_1 * self.K_tau
        self.r_2 = self.r_1 * self.K_r

        self.N_3 = self.N_2 * self.K_N
        self.tau_3 = self.tau_2 * self.K_tau
        self.r_3 = self.r_2 * self.K_r

        self.eps, self.min_samples = int(eps), int(min_samples)
        self.max_slices_for_proto = int(max_slices_for_proto)

        # Train or load HOTS prototypes once for this dataset/config
        proto_cache_path = None
        if prototype_cache_root:
            os.makedirs(prototype_cache_root, exist_ok=True)
            key = f"proto{self.max_slices_for_proto}_N1{self.N_1}_tau1{self.tau_1}_r1{self.r_1}_KN{self.K_N}_Ktau{self.K_tau}_Kr{self.K_r}_HxW{self.height}x{self.width}.npz"
            proto_cache_path = os.path.join(prototype_cache_root, key)

        self.C_1, self.C_2, self.C_3 = self._build_or_load_prototypes(proto_cache_path)

        self.debug_save_l3 = bool(debug_save_l3)
        self.debug_mode = debug_mode
        self.debug_max_slices = debug_max_slices
        self.debug_root = debug_root

    # ---- dataset protocol ----
    def __len__(self):
        return len(self.preprocessor)

    def __getitem__(self, idx):
        item_dict = self.preprocessor[idx]
        events_t, events_xy, events_p, label, path = item_dict['events_t'], item_dict['events_xy'], item_dict['events_p'], item_dict['label'], item_dict['path']

        # cache path per sequence (mirror your TimeSurface layout)
        rel_seq_path = os.path.relpath(path, start=self.dataset_dir)
        cache_dir_path = os.path.join(self.cache_root, rel_seq_path, 'hots')
        cache_name = (
            f"hots_N1{self.N_1}_KN{self.K_N}_Ktau{self.K_tau}_Kr{self.K_r}"
            f"_eps{self.eps}_ms{self.min_samples}.pt"
        )
        cached_path = os.path.join(cache_dir_path, cache_name)

        if self.use_cache and os.path.exists(cached_path):
            return_dict = torch.load(cached_path)
            # print(f"Loaded HOTS features from {cached_path}")
            return return_dict

        # Encode each temporal slice to a single histogram over layer-3 prototypes

        if not self.use_polarity:
            p = np.ones_like(p, dtype=np.int64)

        ev = _arrays_to_event_list(events_t - events_t[0], events_xy, events_p)

        if self.denoise:
            ev_filt = remove_isolated_pixels(ev, eps=self.eps, min_samples=self.min_samples)[0]
        else:
            ev_filt = ev

        # L1 -> L2 -> L3
        ev_l2 = generate_layer_outputs(
            num_polarities=2, features=self.C_1, tau=self.tau_1, r=self.r_1,
            width=self.width, height=self.height, events=ev_filt
        )
        ev_l3 = generate_layer_outputs(
            num_polarities=self.N_1, features=self.C_2, tau=self.tau_2, r=self.r_2,
            width=self.width, height=self.height, events=ev_l2
        )
        ev_out = generate_layer_outputs(
            num_polarities=self.N_2, features=self.C_3, tau=self.tau_3, r=self.r_3,
            width=self.width, height=self.height, events=ev_l3
        )

        # # ----- DEBUG SAVE (raw L3 + visualization) -----
        # if self.debug_save_l3 and (self.debug_max_slices is None or sidx < self.debug_max_slices):
        #     # where to save
        #     rel_seq_path = os.path.relpath(path, start=self.dataset_dir)
        #     base_dir = (self.debug_root if self.debug_root 
        #                 else os.path.join(self.cache_root, rel_seq_path, 'hots_debug'))
        #     pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

        #     # raw arrays
        #     tt, xxyy, pp = _events_to_arrays(ev_out)
        #     npz_path = os.path.join(base_dir, f"l3_events_slice{sidx:04d}.npz")
        #     np.savez_compressed(npz_path, t=tt, xy=xxyy, p=pp)

        #     # label-map visualization
        #     lab_l1 = _l3_events_to_labelmap(ev_l2, self.height, self.width, self.N_1, mode=self.debug_mode)
        #     lab_l2 = _l3_events_to_labelmap(ev_l3, self.height, self.width, self.N_2, mode=self.debug_mode)
        #     lab_l3 = _l3_events_to_labelmap(ev_out, self.height, self.width, self.N_3, mode=self.debug_mode)
        #     png_path_l1 = os.path.join(base_dir, f"l1_labelmap_slice{sidx:04d}.png")
        #     png_path_l2 = os.path.join(base_dir, f"l2_labelmap_slice{sidx:04d}.png")
        #     png_path_l3 = os.path.join(base_dir, f"l3_labelmap_slice{sidx:04d}.png")
        #     _save_labelmap_png(lab_l1, png_path_l1, self.N_1)
        #     _save_labelmap_png(lab_l2, png_path_l2, self.N_2)
        #     _save_labelmap_png(lab_l3, png_path_l3, self.N_3)

        #     print(np.unique(pp, return_counts=True))

        # n0 = len(ev)                    # raw
        # n1 = len(ev_filt)               # after noise filter
        # n2 = len(ev_l2)
        # n3 = len(ev_l3)
        # n4 = len(ev_out)
        # if n4 == 0:
        #     print(f"[slice {sidx}] empty: raw={n0}, filt={n1}, L2={n2}, L3={n3}, Lout={n4}")

        # ----- Histogram feature (your original) -----
        h = _histogram_over_prototypes(ev_out, n_bins=self.N_3)
        h = normalize_array(h, self.normalize)

        feats = torch.from_numpy(h).float()

        return_dict = {
            'data': feats,
            'label': label,
            'path': path
        }

        if self.use_cache:
            os.makedirs(cache_dir_path, exist_ok=True)
            torch.save(return_dict, cached_path)
            # print(f"Saved HOTS features to {cached_path}")

        return return_dict

    # ---- same padded collate contract as your TimeSurface class ----
    @staticmethod
    def collate_fn(batch):
        sequences, labels = zip(*batch)  # each is (L, D)
        valid_len = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
        padded_sequences = pad_sequence(sequences, batch_first=True)  # (B, Lmax, D)
        return padded_sequences, valid_len, torch.stack(labels)

    # ---- prototype training exactly via repo functions ----
    def _build_or_load_prototypes(self, cache_path: str | None):
        print("Cache path:", cache_path)
        if cache_path and os.path.isfile(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            print(f"Loaded HOTS prototypes from {cache_path}")
            print(data["C_1"].shape, data["C_2"].shape, data["C_3"].shape)
            return data["C_1"], data["C_2"], data["C_3"]

        # Build a pooled, concatenated (filtered) event list from up to max_slices_for_proto
        ev_all_filt = []
        ts_offset = 0
        used = 0

        for i in range(len(self.preprocessor)):
            if used >= self.max_slices_for_proto:
                break
            item = self.preprocessor[i]

            print(f"Collecting slice {used} for prototype training...")
            if used >= self.max_slices_for_proto:
                break
            if not self.use_polarity:
                p = np.ones_like(p, dtype=np.int64)

            ev = _arrays_to_event_list(item['events_t'] - item['events_t'][0], item['events_xy'], item['events_p'])

            # shift timestamps to keep monotonicity across concatenation (same idea as repo)
            if ev_all_filt:
                for e in ev:
                    e.ts += ts_offset
            if len(ev):
                ts_offset = max(ts_offset, ev[-1].ts)

            if self.denoise:
                ev_filt = remove_isolated_pixels(ev, eps=self.eps, min_samples=self.min_samples)[0]
                print(f"  raw events: {len(ev)}, filtered: {len(ev_filt)}")
            else:
                ev_filt = ev
            
            ev_all_filt.extend(ev_filt)
            used += 1

        if not ev_all_filt:
            raise RuntimeError("No events found to train prototypes. Check Preprocessor outputs.")
        
        print(f"Training HOTS prototypes on {used} slices, {len(ev_all_filt)} total events...")

        # ----- Train Layer 1 -----
        C_1 = initialise_time_surface_prototypes(
            self.N_1, self.tau_1, self.r_1, self.width, self.height, ev_all_filt, plot=False
        )
        train_layer(
            C_1, self.N_1, self.tau_1, self.r_1, self.width, self.height,
            ev_all_filt, num_polarities=2, layer_number=1, plot=False
        )
        print("Trained layer 1 prototypes.")

        # ----- Train Layer 2 -----
        ev_l2 = generate_layer_outputs(
            num_polarities=2, features=C_1, tau=self.tau_1, r=self.r_1,
            width=self.width, height=self.height, events=ev_all_filt
        )
        C_2 = initialise_time_surface_prototypes(
            self.N_2, self.tau_2, self.r_2, self.width, self.height, ev_l2, plot=False
        )
        train_layer(
            C_2, self.N_2, self.tau_2, self.r_2, self.width, self.height,
            ev_l2, num_polarities=self.N_1, layer_number=2, plot=False
        )
        print("Trained layer 2 prototypes.")

        # ----- Train Layer 3 -----
        ev_l3 = generate_layer_outputs(
            num_polarities=self.N_1, features=C_2, tau=self.tau_2, r=self.r_2,
            width=self.width, height=self.height, events=ev_l2
        )
        C_3 = initialise_time_surface_prototypes(
            self.N_3, self.tau_3, self.r_3, self.width, self.height, ev_l3, plot=False
        )
        train_layer(
            C_3, self.N_3, self.tau_3, self.r_3, self.width, self.height,
            ev_l3, num_polarities=self.N_2, layer_number=3, plot=False
        )
        print("Trained layer 3 prototypes.")

        if cache_path:
            np.savez_compressed(cache_path, C_1=C_1, C_2=C_2, C_3=C_3)

        return C_1, C_2, C_3


# ---------- Example main (mirrors yours) ----------
def main():
    import time
    dataset_dir = "/fs/nexus-projects/DVS_Actions/NatureRoboticsData/"

    dataset = HOTS(
        dataset_dir=dataset_dir,
        width=680, height=680,
        use_cache=False,                         # recompute to emit debug files
        cache_root="/fs/nexus-projects/DVS_Actions/NatureRoboticsDataCache",
        use_polarity=True,
        purpose="train",
        N_1=2, tau_1=20000.0, r_1=10,
        K_N=2, K_tau=2.0, K_r=4,
        prototype_cache_root="./cache/hots_prototypes",
        max_slices_for_proto=2,
    )

    dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)  # batch_size=1 -> no padding needed
    it = iter(dl)
    for _ in tqdm(range(len(dl)), desc="Batches"):
        t0 = time.perf_counter()
        feats, label = next(it)
        dt = time.perf_counter() - t0
        print("HOTS features:", feats.shape, "label:", label, "dt:", f"{dt:.4f}s")
        break


if __name__ == "__main__":
    main()
