import torch
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced, TrajSlicerDataset
from einops import rearrange

import numpy as np
import decord
import pickle
from pathlib import Path
from decord import VideoReader
from typing import Optional, Callable, Any


decord.bridge.set_bridge("torch")

# mean and var after dividing by 100, without considering the filling 0 actions
# ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
# ACTION_STD = torch.tensor([0.2019, 0.2002])

# STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584])
# STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556])

# PROPRIO_MEAN = torch.tensor([236.6155, 264.5674])
# PROPRIO_STD = torch.tensor([101.1202, 87.0112])

# calculated from 185 train trajs
# ACTION_MEAN = torch.tensor([-0.007838 ,  0.0068466])
# ACTION_STD = torch.tensor([0.20135775, 0.1994587])

# STATE_MEAN = torch.tensor([228.79599  , 292.14444  , 246.62141  , 271.63977  ,   1.8079071, -2.93032027,  2.54307914])
# STATE_STD = torch.tensor([101.7689   ,  97.10681  ,  64.57505  ,  66.53813  ,   1.7803417, 74.84556075, 74.14009094])

# PROPRIO_MEAN = torch.tensor([228.79599  , 292.14444, -2.93032027,  2.54307914])
# PROPRIO_STD = torch.tensor([101.7689   ,  97.10681, 74.84556075, 74.14009094])

# note: velocity stats are calculated separately! other stats are unchanged due to compatibility with prev models
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])

STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])

PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])

class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "/data/datasets/pusht_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
        state_based: bool = False,
        with_velocity: bool = True,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]

        self.proprios = self.states[..., :2].clone()  # first 2 dim of states is proprio

        self.with_velocity = with_velocity # TODO: add vel in proprios
        if with_velocity:
            with open(self.data_path / "velocities.pkl", "rb") as f:
                self.velocities = torch.from_numpy(pickle.load(f))
                self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[:self.state_dim]  # if normalize_actions and state_based, assume normalize states as well
            self.state_std = STATE_STD[:self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
            self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

        self.state_based = state_based

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        mask = torch.ones(len(act)).bool()
        if not self.state_based:
            image = reader.get_batch(frames)  # THWC
            image = image / 255.0
            image = rearrange(image, "T H W C -> T C H W")
            if self.transform:
                image = self.transform(image)
            obs = {"visual": image, "proprio": proprio}
        else:
            state = (state - self.state_mean) / self.state_std
            obs = {"visual": state, "proprio": proprio}

        return obs, act, state, mask

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_pusht_slice_train_val(
    transform,
    n_rollout=50,
    data_path="/data/datasets/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    state_based=False,
    with_velocity=True,
):
    train_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        state_based=state_based,
        with_velocity=with_velocity,
    )
    val_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        state_based=state_based,
        with_velocity=with_velocity,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)
    # dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
    #     traj_dataset=dset,
    #     train_fraction=split_ratio,
    #     num_frames=num_hist + num_pred,
    #     frameskip=frameskip
    # )

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset


if __name__ == "__main__":
    from torchvision import datasets, transforms, utils

    img_size = 224
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dset = PushTDataset(
        transform=None,
        relative=True,
        action_scale=100.0,
    )

    import gym
    from gym.envs.registration import register

    register(
        id="pusht",
        entry_point="pusht_env:PushTEnv",
        max_episode_steps=300,
        reward_threshold=1.0,
    )
    env = gym.make("pusht")
    observations, act, state, mask = dset[0]
    act = act * ACTION_STD + ACTION_MEAN
    env.env.env.reset_to_state = state[0].numpy()
    obs0 = env.reset()
    import numpy as np

    first_obs = np.concatenate(
        [observations[0].permute(1, 2, 0).numpy(), obs0[0]], axis=1
    )
    frames = [first_obs]
    # import pdb; pdb.set_trace()
    for i in range(len(act) - 1):
        obs, reward, done, info = env.step(act[i])
        # import pdb; pdb.set_trace()
        original_obs = observations[i + 1].permute(1, 2, 0).numpy()
        # stitch the two images together
        obs = np.concatenate([original_obs, obs], axis=1)
        frames.append(obs)
    # save all frames to a video
    import imageio
    import os

    video_writer = imageio.get_writer("output_without_action_scale.mp4", fps=10)
    for image in frames:
        video_writer.append_data(image)
    video_writer.close()

    import pdb

    pdb.set_trace()

    train_dset, val_dset = get_train_val_sliced(dset, train_fraction=0.5, num_frames=4)

    dataloader = torch.utils.data.DataLoader(
        train_dset, batch_size=7, shuffle=True, num_workers=1, collate_fn=None
    )

    for i, data in enumerate(dataloader):
        # data: [b, num_hist + num_pred, 3, img_size, img_size]
        import pdb

        pdb.set_trace()

# def get_push_t_train_val(
#     data_path,
#     subset_fraction: Optional[float] = None,
#     relative: bool = True,
#     action_scale=100.0,
#     train_fraction=0.9,
#     random_seed=42,
#     window_size=10,
#     goal_conditional: Optional[str] = None,
#     future_seq_len: Optional[int] = None,
#     min_future_sep: int = 0,
#     transform: Optional[Callable[[Any], Any]] = None,
# ):
#     if goal_conditional is not None:
#         assert goal_conditional in ["future"]
#     return get_train_val_sliced(
#         PushTDataset(
#             data_path,
#             subset_fraction=subset_fraction,
#             relative=relative,
#             action_scale=action_scale,
#         ),
#         train_fraction,
#         random_seed,
#         window_size,
#         future_conditional=(goal_conditional == "future"),
#         min_future_sep=min_future_sep,
#         future_seq_len=future_seq_len,
#         transform=transform,
#     )
