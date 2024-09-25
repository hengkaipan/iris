
import torch
from typing import Callable, Optional
from .traj_dset import TrajDataset
from einops import rearrange
from tqdm import tqdm
import sys
sys.path.append('/home/kathy/dev/robohive/')
from robohive.logger.grouped_datasets import Trace

class RoboHiveTrajDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_file: str = "/home/kathy/dev/robohive/robohive/datasets/DAPG_expert/door_v2d-v1/door_v2d-v120230407-061545_trace.h5",
        preload: bool = False,
    ):
        self.camera_name = "rgb:view_1:224x224:2d"
        self.transform = transform
        self.preload = preload
        self.path_observations = []
        self.path_actions = []

        paths = Trace.load(trace_path=data_file, trace_type="RoboHive")
        self.paths = paths
        self.action_dim = paths[0]["actions"].shape[-1]

        if n_rollout is not None:
            self.size = min(
                n_rollout, len(paths)
            )  # TODO: hack, check n_rollout < len(paths)
        else:
            self.size = len(paths)

        if self.preload:
            for i in tqdm(range(self.size), desc="Loading Data"):
                path = paths[i]
                obs = path["env_infos"]["visual_dict"][self.camera_name][...]
                obs = torch.tensor(self.preprocess_imgs(obs))
                if self.transform is not None:
                    obs = self.transform(obs).type(torch.FloatTensor)
                self.path_observations.append(obs)
                act = path["actions"][...]
                act = torch.tensor(act).type(torch.FloatTensor)
                self.path_actions.append(act)
        print("Finished loading data.")

        # TODO: add observation dim too
        # print("robohive processing finished.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # print(f"Robohive getting item {idx}")
        if self.preload:
            obs = self.path_observations[idx]
            act = self.path_actions[idx]
        else:
            path = self.paths[idx]
            obs = path["env_infos"]["visual_dict"][self.camera_name][
                ...
            ]  # (T, h, w, c)
            obs = torch.tensor(self.preprocess_imgs(obs))
            if self.transform is not None:
                obs = self.transform(obs).type(torch.FloatTensor)
            act = path["actions"][...]  # (T, action_dim)
            act = torch.tensor(act).type(torch.FloatTensor)
        # hack to avoid robohive dset nan issue
        act = act[:-1]  # (T-1, action_dim)
        obs = obs[:-1]  # (T-1, 3, img_size, img_size)
        if torch.isnan(act).any():
            raise ValueError(f"act contains nan at idx {idx}")
        return tuple([obs, act])

    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        return self.paths[idx]["actions"].shape[0] - 1

    def preprocess_imgs(self, imgs):
        """
        Reshape imgs from to (T, h, w, c) to (T, c, h, w)
        """
        imgs = imgs.transpose(0, 3, 1, 2) / 255.0
        return imgs