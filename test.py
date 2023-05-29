# from math import log2
# for signal_size in [7, 14, 28, 56, 112, 224, 401]:
#     num_steps = int(log2(signal_size / 4))
#     print(f'Curr sig size: {signal_size} \n #### num steps: {num_steps}')

# import torch
# from math import log2

# real = torch.randn((48, 1, 401))
# desired_steps = 6
# signal_sizes = []
# scale_factor = []

# for steps in range(desired_steps + 1):
#     if steps == 0:
#         signal_sizes.append(real.shape[2])
#     else:
#         signal_sizes.append(signal_sizes[steps-1] // 2)

# signal_sizes.sort()
# print(signal_sizes)


# last_signal_size = 0
# for signal_size in signal_sizes:
#     if not signal_size == signal_sizes[0]:
#         scale_factor.append(signal_size / last_signal_size)
#     last_signal_size = signal_size

# print(scale_factor)
# # for sigal_size in signal_sizes:
# #     print(f'{sigal_size}: {int(log2(signal_size / 4))}')
# for idx, signal_size in enumerate(signal_sizes):
#     print(f'sigal_size: {signal_size}, idx: {idx}')

# desired_steps = 6
# factors = [1, 1, 1, 1, 1, 1, 1]
# factors2 = [1 for _ in range(desired_steps + 1)]
# assert len(factors) == len(factors2)
# for i in factors:
#     assert factors[i] == factors2[i]
# print('success')

#####################################

# import os 
# print(os.path.join(os.getcwd(), "resources\data"))

#####################################

# BATCH_SIZES = [32, 16, 16, 16, 16, 8, 4]
# PROGRESSIVE_EPOCHS = [2000] * len(BATCH_SIZES)
# print(PROGRESSIVE_EPOCHS)
# print(BATCH_SIZES[1:])


import os
import numpy as np
import importlib
import mne
from typing import List, Tuple

mne.set_log_level("error")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

MAPPING = {7: "feet", 8: "left_hand", 9: "right_hand", 10: "tongue"}

cfg = dict(
    preprocessing=dict(
        target_freq=100,
        low_freq=8,
        high_freq=25,
        average_ref=True,
        normalize = True
    ),
    epochs=dict(
        baseline=(-0.1, 1.9),
        tmin=-0.1,
        tmax=5.9,
    )
)

class MI_Dataset_ALL(Dataset):
    def __init__(self, data_folder="data", subject_ids=list(range(1, 10)), signals=None, device="cpu", config="default", verbose=False, transform=None, flatten=False):
        self.data_root = data_folder
        self.subject_ids = subject_ids
        self.device = device

        self.signals = signals if signals is not None else list(MAPPING.values())  # Include all signals by default
        self.transform = transform
        self.flatten = flatten

        self.load_config()
        self.load_raw()
        self.apply_preprocess()
        self.create_epochs()
        if verbose:
            print(self.epochs)

        self.format_data()

        self.time_steps = self.X.shape[-1]
        self.channels = self.X.shape[-2]
        
        if verbose:
            print("#" * 50)
            print("Dataset created:")
            print(f"X --> {self.X.shape} ({self.X.dtype})")
            print(f"y --> {self.y.shape} ({self.y.dtype})")
            print("#" * 50)

    def load_config(self) -> None:

        self.target_freq = cfg["preprocessing"]["target_freq"]
        self.low_freq = cfg["preprocessing"]["low_freq"]
        self.high_freq = cfg["preprocessing"]["high_freq"]
        self.average_ref = cfg["preprocessing"]["average_ref"]

        self.baseline = cfg["epochs"]["baseline"]
        self.tmin = cfg["epochs"]["tmin"]
        self.tmax = cfg["epochs"]["tmax"]

        self.normalize = cfg["preprocessing"]["normalize"]

    def load_raw(self) -> None:
        self.subject_paths = [
            os.path.join(self.data_root, "A0" + str(subject_id) + "T.gdf")
            for subject_id in self.subject_ids
        ]
        
        self.raws = [
            mne.io.read_raw_gdf(subject_path, preload=True)
            for subject_path in self.subject_paths
        ]

        # channels_to_keep = ["EEG-Cz", "EEG-C4", "EEG-C3"]
        
        # for raw in self.raws:
        #     raw.pick_channels(channels_to_keep)

        # Only needed if pick_channels isn't used
        for raw in self.raws:
            eog_channels = [
                i for i, ch_name in enumerate(raw.ch_names) if "EOG" in ch_name
            ]
            raw.drop_channels([raw.ch_names[ch_idx] for ch_idx in eog_channels])

        self.filter_events()

    # def filter_events(self) -> None:
    #     for raw in self.raws:
    #         events, event_ids = mne.events_from_annotations(raw)
    #         desired_ids = {k: v for k, v in MAPPING.items() if v in self.signals}  # Filter the event_ids by signals
    #         # desired_ids = {'right_hand': 9}
    #         # filtered_ids = {k: v for k, v in desired_ids.items() if v in event_ids}
    #         key = next(iter(desired_ids.keys()))
    #         found = False
    #         for name, idx in event_ids.items():
    #             if idx == key:
    #                 found = True

    #         if found:
    #             annot_from_events = mne.annotations_from_events(
    #                 events, event_desc=desired_ids, sfreq=raw.info["sfreq"]
    #             )
    #             raw.set_annotations(annot_from_events)


    def filter_events(self) -> None:
        for raw in self.raws:
            events, _ = mne.events_from_annotations(raw)
            event_ids = {k: v for k, v in MAPPING.items() if v in self.signals}  # Filter the event_ids by signals
            
            annot_from_events = mne.annotations_from_events(
                events, event_desc=event_ids, sfreq=raw.info["sfreq"]
            )

            raw.set_annotations(annot_from_events)
            
    def apply_preprocess(self) -> None:
        def preprocess_raw(session):
            session = session.resample(self.target_freq, npad="auto")
            if self.average_ref:
                session = session.set_eeg_reference("average", projection=True)
            session = session.filter(l_freq=self.low_freq, h_freq=self.high_freq)
            return session

        self.raws = [preprocess_raw(raw) for raw in self.raws]

    def create_epochs(self):
        def split2epochs(session):
            events, event_ids = mne.events_from_annotations(session)
            return mne.Epochs(
                session,
                events=events,
                event_id=event_ids,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=self.baseline,
                preload=True,
            )

        self.epochs = [split2epochs(raw) for raw in self.raws]
        self.epochs = mne.concatenate_epochs(self.epochs)
        self.epochs = self.epochs.crop(tmin=self.baseline[-1], tmax=self.tmax)
        #self.epochs.plot()
        del self.raws

    def format_data(self):
        # self.X = self.epochs.get_data()
        self.X = self.epochs.get_data()[:, :, :400]

        self.y = self.epochs.events[:, -1]
        self.y -= 1  # start at 0

        if self.transform:
            self.X, self.y = self.transform(self.X, self.y)

        if self.normalize:
            self.do_normalize()
        
        if self.flatten:
            self.X = self.X.reshape(-1, 1, self.X.shape[-1])
            self.y = np.repeat(self.y, 22)

        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def do_normalize(self):
        orig_shape = self.X.shape
        self.X = self.X.reshape(self.X.shape[0], -1)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.X = self.X.reshape(orig_shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

from scipy import signal
from torch.utils.data import DataLoader

class ResizeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, X, y):
        old_size = X.shape[2]
        X_resized = signal.resample(X, self.new_size, axis=2)
        return X_resized, y

def main():
    transform = ResizeTransform(6)
    right_hand_subjects = [1,2,3,5,6,7,8,9]
    dataset = MI_Dataset_ALL("resources/data", subject_ids=[1,2,3,5,6,7,8,9], signals=["right_hand"], device="cpu", verbose=True)
    # loader = DataLoader(dataset, batch_size=96, shuffle=True, num_workers=4)



if __name__ == '__main__':
    main()

