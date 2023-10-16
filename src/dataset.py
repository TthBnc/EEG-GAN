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

class MI_Dataset(Dataset):
    def __init__(self, subject_id, runs, signals=None, device="cpu", config="default", verbose=False, transform=None, flatten=False):
        self.subject_id = subject_id
        self.device = device
        self.runs = runs
        self.signals = signals if signals is not None else list(MAPPING.values())
        self.transform = transform
        self.flatten = flatten
        self.load_config()
        self.load_raw()
        self.apply_preprocess()
        self.create_epochs()
        if verbose:
            print(self.epochs)

        self.split_by_runs()
        self.format_data()

        self.time_steps = self.X.shape[-1]
        self.channels = self.X.shape[-2]
        
        if verbose:
            print("#" * 50)
            print("Dataset created:")
            print(f"X --> {self.X.shape} ({self.X.dtype})")
            print(f"y --> {self.y.shape} ({self.y.dtype})")
            print("#" * 50)

    def load_config(self):

        self.target_freq = cfg["preprocessing"]["target_freq"]
        self.low_freq = cfg["preprocessing"]["low_freq"]
        self.high_freq = cfg["preprocessing"]["high_freq"]
        self.average_ref = cfg["preprocessing"]["average_ref"]

        self.baseline = cfg["epochs"]["baseline"]
        self.tmin = cfg["epochs"]["tmin"]
        self.tmax = cfg["epochs"]["tmax"]

        self.normalize = cfg["preprocessing"]["normalize"]

    def load_raw(self):
        subject_path =  "A0" + str(self.subject_id) + "T.gdf"
        self.raw = mne.io.read_raw_gdf(subject_path, preload=True)

        # # Keep only desired channels
        # channels_to_keep = ["EEG-Cz", "EEG-C4", "EEG-C3"]
        # # self.raw.pick_channels(channels_to_keep)
        # for raw in self.raws:
        #     raw.pick_channels(channels_to_keep)

        # Not needed if .pick_channels is used
        eog_channels = [
            i for i, ch_name in enumerate(self.raw.ch_names) if "EOG" in ch_name
        ]
        self.raw.drop_channels([self.raw.ch_names[ch_idx] for ch_idx in eog_channels])

        self.filter_events()

    def filter_events(self):
        events, _ = mne.events_from_annotations(self.raw)
        event_ids = {k: v for k, v in MAPPING.items() if v in self.signals}
        annot_from_events = mne.annotations_from_events(
            events, event_desc=event_ids, sfreq=self.raw.info["sfreq"]
        )

        self.raw.set_annotations(annot_from_events)

    def apply_preprocess(self):
        self.raw = self.raw.resample(self.target_freq, npad="auto")
        if self.average_ref:
            self.raw = self.raw.set_eeg_reference("average", projection=True)

        self.raw = self.raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)

    def create_epochs(self):
        events, event_ids = mne.events_from_annotations(self.raw)
        self.epochs = mne.Epochs(
            self.raw,
            events=events,
            event_id=event_ids,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            preload=True,
        )
        self.epochs = self.epochs.crop(tmin=self.baseline[-1], tmax=self.tmax)
        # self.epochs.plot()
        del self.raw

    def split_by_runs(self):
        # X = self.epochs.get_data()
        self.X = self.epochs.get_data()[:, :, :400]

        if self.transform:
            self.X, self.y = self.transform(self.X, self.y)

        if self.normalize:
            orig_shape = X.shape
            X = X.reshape(X.shape[0], -1)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X = X.reshape(orig_shape)
            y = self.epochs.events[:, -1]
        y -= 1  # start at 0

        if self.flatten:
            self.X = self.X.reshape(-1, 1, self.X.shape[-1])

        X_by_runs = []
        y_by_runs = []

        for index in range(0, int(X.shape[0] // 48)):
            X_by_runs.append(X[index * 48 : (index + 1) * 48])
            y_by_runs.append(y[index * 48 : (index + 1) * 48])

        self.runs_features = np.array(X_by_runs)
        self.runs_labels = np.array(y_by_runs)

        max_run_index = len(self.runs_features) - 1

        self.runs = [run for run in self.runs if run <= max_run_index]

        if not self.runs:
            raise ValueError("No valid runs available after filtering.")

        self.runs_features = self.runs_features[self.runs]
        self.runs_labels = self.runs_labels[self.runs]

        self.X = self.runs_features.reshape(
            -1, self.runs_features.shape[2], self.runs_features.shape[3]
        )
        self.y = self.runs_labels.reshape(-1)

    

    def format_data(self):
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



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
    #         events, _ = mne.events_from_annotations(raw)
    #         unique_raw_event_ids = np.unique(events[:, -1])
    #         event_ids = {k: v for k, v in MAPPING.items() if v in self.signals}  # Filter the event_ids by signals
    #         if any(id in unique_raw_event_ids for id in event_ids):
    #             annot_from_events = mne.annotations_from_events(
    #                 events, event_desc=event_ids, sfreq=raw.info["sfreq"]
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
        # self.y = np.array([signal for signal, signal_id in self.epochs.event_id.items() if signal_id in self.y])


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
