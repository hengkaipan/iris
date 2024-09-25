import torch
from .traj_dset import TrajDataset


class AlternatingDataset(TrajDataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = min(len(dataset1), len(dataset2)) * 2

        self.proprio_dim = max(dataset1.proprio_dim, dataset2.proprio_dim)
        self.state_dim = max(dataset1.state_dim, dataset2.state_dim)
        self.action_dim = max(dataset1.action_dim, dataset2.action_dim)

    def __len__(self):
        return self.length

    def pad_data(self, data, target_dim):
        if data.shape[-1] < target_dim:
            pad_size = target_dim - data.shape[-1]
            padding = (0, pad_size)  # Pad on the last dimension
            data = torch.nn.functional.pad(data, padding)
        return data

    def __getitem__(self, idx):
        if idx % 2 == 0:
            data = self.dataset1[idx // 2]
        else:
            data = self.dataset2[idx // 2]

        obs, act, state, mask = data

        act = self.pad_data(act, self.action_dim)
        state = self.pad_data(state, self.state_dim)

        if "proprio" in obs:
            obs["proprio"] = self.pad_data(obs["proprio"], self.proprio_dim)

        return obs, act, state, mask

    def get_seq_length(self, idx):
        if idx % 2 == 0:
            return self.dataset1.get_seq_length(idx // 2)
        else:
            return self.dataset2.get_seq_length(idx // 2)
