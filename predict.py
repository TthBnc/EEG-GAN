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

# MAPPING = {7: "feet", 8: "left_hand", 9: "right_hand", 10: "tongue"}

# cfg = dict(
#     preprocessing=dict(
#         target_freq=100,
#         low_freq=8,
#         high_freq=25,
#         average_ref=True,
#         normalize = True
#     ),
#     epochs=dict(
#         baseline=(-0.1, 1.9),
#         tmin=-0.1,
#         tmax=5.9,
#     )
# )

# class MI_Dataset_ALL(Dataset):
#     def __init__(self, data_folder="data", subject_ids=list(range(1, 10)), signals=None, device="cpu", config="default", verbose=False, transform=None, flatten=False):
#         self.data_root = data_folder
#         self.subject_ids = subject_ids
#         self.device = device

#         self.signals = signals if signals is not None else list(MAPPING.values())  # Include all signals by default
#         self.transform = transform
#         self.flatten = flatten

#         self.load_config()
#         self.load_raw()
#         self.apply_preprocess()
#         self.create_epochs()
#         if verbose:
#             print(self.epochs)

#         self.format_data()

#         self.time_steps = self.X.shape[-1]
#         self.channels = self.X.shape[-2]
        
#         if verbose:
#             print("#" * 50)
#             print("Dataset created:")
#             print(f"X --> {self.X.shape} ({self.X.dtype})")
#             print(f"y --> {self.y.shape} ({self.y.dtype})")
#             print("#" * 50)

#     def load_config(self) -> None:

#         self.target_freq = cfg["preprocessing"]["target_freq"]
#         self.low_freq = cfg["preprocessing"]["low_freq"]
#         self.high_freq = cfg["preprocessing"]["high_freq"]
#         self.average_ref = cfg["preprocessing"]["average_ref"]

#         self.baseline = cfg["epochs"]["baseline"]
#         self.tmin = cfg["epochs"]["tmin"]
#         self.tmax = cfg["epochs"]["tmax"]

#         self.normalize = cfg["preprocessing"]["normalize"]

#     def load_raw(self) -> None:
#         self.subject_paths = [
#             os.path.join(self.data_root, "A0" + str(subject_id) + "T.gdf")
#             for subject_id in self.subject_ids
#         ]
        
#         self.raws = [
#             mne.io.read_raw_gdf(subject_path, preload=True)
#             for subject_path in self.subject_paths
#         ]

#         # channels_to_keep = ["EEG-Cz", "EEG-C4", "EEG-C3"]
        
#         # for raw in self.raws:
#         #     raw.pick_channels(channels_to_keep)

#         # Only needed if pick_channels isn't used
#         for raw in self.raws:
#             eog_channels = [
#                 i for i, ch_name in enumerate(raw.ch_names) if "EOG" in ch_name
#             ]
#             raw.drop_channels([raw.ch_names[ch_idx] for ch_idx in eog_channels])

#         self.filter_events()

#     def filter_events(self) -> None:
#         for raw in self.raws:
#             events, _ = mne.events_from_annotations(raw)
#             event_ids = {k: v for k, v in MAPPING.items() if v in self.signals}  # Filter the event_ids by signals
#             annot_from_events = mne.annotations_from_events(
#                 events, event_desc=event_ids, sfreq=raw.info["sfreq"]
#             )

#             raw.set_annotations(annot_from_events)

#     def apply_preprocess(self) -> None:
#         def preprocess_raw(session):
#             session = session.resample(self.target_freq, npad="auto")
#             if self.average_ref:
#                 session = session.set_eeg_reference("average", projection=True)
#             session = session.filter(l_freq=self.low_freq, h_freq=self.high_freq)
#             return session

#         self.raws = [preprocess_raw(raw) for raw in self.raws]

#     def create_epochs(self):
#         def split2epochs(session):
#             events, event_ids = mne.events_from_annotations(session)
#             return mne.Epochs(
#                 session,
#                 events=events,
#                 event_id=event_ids,
#                 tmin=self.tmin,
#                 tmax=self.tmax,
#                 baseline=self.baseline,
#                 preload=True,
#             )

#         self.epochs = [split2epochs(raw) for raw in self.raws]
#         self.epochs = mne.concatenate_epochs(self.epochs)
#         self.epochs = self.epochs.crop(tmin=self.baseline[-1], tmax=self.tmax)
#         #self.epochs.plot()
#         del self.raws

#     def format_data(self):
#         # self.X = self.epochs.get_data()
#         self.X = self.epochs.get_data()[:, :, :400]

#         self.y = self.epochs.events[:, -1]
#         self.y -= 1  # start at 0

#         self.X = np.mean(self.X, axis=1, keepdims=True)

#         if self.transform:
#             self.X, self.y = self.transform(self.X, self.y)

#         if self.normalize:
#             self.do_normalize()
        
#         if self.flatten:
#             self.X = self.X.reshape(-1, 1, self.X.shape[-1])
#             self.y = np.repeat(self.y, 22)

#         self.X = torch.from_numpy(self.X).float()
#         self.y = torch.from_numpy(self.y).long()

#         self.X = self.X.to(self.device)
#         self.y = self.y.to(self.device)

#     def do_normalize(self):
#         orig_shape = self.X.shape
#         self.X = self.X.reshape(self.X.shape[0], -1)
#         scaler = StandardScaler()
#         self.X = scaler.fit_transform(self.X)
#         self.X = self.X.reshape(orig_shape)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


import torch
from model import EEG_PRO_GAN_Generator
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


fixed_noise = torch.randn(5000, 200, 1).to("cpu")
real = torch.randn((48, 1, 400))
DESIRED_STEPS = 6
FACTORS = [1 for _ in range(DESIRED_STEPS + 1)]
SIGNAL_SIZES = []
SCALE_FACTORS = []

for steps in range(DESIRED_STEPS + 1):
    if steps == 0:
        SIGNAL_SIZES.append(real.shape[2])
    else:
        SIGNAL_SIZES.append(SIGNAL_SIZES[steps-1] // 2)

SIGNAL_SIZES.sort()

last_signal_size = 0
for signal_size in SIGNAL_SIZES:
    if not signal_size == SIGNAL_SIZES[0]:
        SCALE_FACTORS.append(signal_size / last_signal_size)
    last_signal_size = signal_size

signal_to_label = {'feet': 1, 'left_hand': 2, 'right_hand': 3, 'tongue': 4}


# dataset = MI_Dataset_ALL("resources/data", subject_ids=[1,2,3,5,6,7,8,9], signals=["right_hand"], verbose=True)
# loader = DataLoader(dataset, batch_size=48, shuffle=True)

gen = EEG_PRO_GAN_Generator(200, 50, SIGNAL_SIZES[0], factors=FACTORS, signal_channels=1).to("cpu")
checkpoint = torch.load("models/one_channel/tongue/tongue_generator_checkpoint_6_20.pth", map_location=torch.device('cpu'))
gen.load_state_dict(checkpoint["model_state"])
gen.eval()
fake = gen(fixed_noise, alpha=0.5, steps=6, scale_factors=SCALE_FACTORS)
# fake = torch.squeeze(fake)
# for batch_idx, (real, _) in enumerate(loader):
#     real = real.to("cpu")
#     fig, axs = plt.subplots(8)
#     fig.suptitle('Real and Fake EEG signals')
#     for i in range(4):
#         axs[i].plot(real[i, 0, :].cpu().detach().numpy())
#         axs[i].set_title(f"Real Channel {i+1}")
#         axs[i+4].plot(fake[i, 0, :].cpu().detach().numpy())
#         axs[i+4].set_title(f"Fake Channel {i+1}")
#     # plt.plot(fake[0, 1, :].cpu().detach().numpy())
#     plt.show()
#     print(fake)
#     break
nparray = fake.detach().numpy()
print(nparray.shape)
np.save('tongue.npy', nparray)


# fake = np.load('left_hand.npy')
# fig, axs = plt.subplots(4)
# fig.suptitle('Fake EEG signals')
# for i in range(4):
#     axs[i].plot(fake[i, 0, :])
#     axs[i].set_title(f'Fake channel {i+1}')
# plt.show()

print("done")