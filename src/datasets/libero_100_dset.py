import os
import h5py
import torch
import bisect
import decord
from tqdm import tqdm
import numpy as np
from typing import Callable, Optional, List
from .traj_dset import _accumulate, TrajSlicerDataset, TrajDataset
from libero.libero import benchmark
from utils import pil_loader
from pathlib import Path
from decord import VideoReader
from einops import rearrange

decord.bridge.set_bridge("torch")

NUM_PATHS_PER_TASK = 50
class Libero100TrajDataset(TrajDataset):
    def __init__(
            self, 
            n_rollout_per_task: Optional[int] = 50,
            transform: Optional[Callable] = None,
            data_path: str = '/data/datasets/libero',
            preload: bool = False,
            task_name: str = 'libero_100',
            tasks: Optional[List[int]] = None, # use all tasks if not specifying tasks
        ):

        self.num_paths = []
        self.camera_name = 'agentview_rgb'
        self.transform = transform
        self.preload = preload # should always be false for consistency
        self.path_observations = []
        self.path_actions = []
        self.path_states = []
        benchmark_dict = benchmark.get_benchmark_dict()
        assert task_name == 'libero_100'

        benchmark_instance_90 = benchmark_dict['libero_90']()
        num_tasks_90 = benchmark_instance_90.get_num_tasks()
        demo_files_90 = [benchmark_instance_90.get_task_demonstration(i) for i in range(num_tasks_90)]
    
        benchmark_instance_10 = benchmark_dict['libero_10']()
        num_tasks_10 = benchmark_instance_10.get_num_tasks()
        demo_files_10 = [benchmark_instance_10.get_task_demonstration(i) for i in range(num_tasks_10)]

        demo_files = demo_files_90 + demo_files_10
        self.demo_files = [
            os.path.join(data_path, file) for file in demo_files
        ]
        num_tasks = num_tasks_90 + num_tasks_10
    
        # get num trajs for each task in tasks
        for t in range(num_tasks):
            demo_file = self.demo_files[t]
            with h5py.File(demo_file, "r") as f:
                if tasks is None or t in tasks:
                    self.num_paths.append(len(f['data']))
                else:
                    self.num_paths.append(0)

        if n_rollout_per_task < NUM_PATHS_PER_TASK:
            self.num_paths = (np.array(self.num_paths) // NUM_PATHS_PER_TASK) * n_rollout_per_task
            self.size = sum(self.num_paths)
        else:
            self.size = sum(self.num_paths)
        
        self.data_path = data_path
        self.task_name = task_name
        self.processed_name = f'{task_name}_processed'
        self.subtask_names = [x.split('/')[-1].split('.')[0][:-5] for x in demo_files] # remove '_demo'

        # always preload actions and init_states, but only preload obs if self.preload
        for t in tqdm(range(num_tasks), desc="Loading Data"):
            demo_file = self.demo_files[t]
            with h5py.File(demo_file, "r") as f:
                observations = []
                actions = []
                states = []
                for j in range(self.num_paths[t]): # load the first n trajs
                    act = f[f'data/demo_{j}']['actions'][...]
                    act = torch.tensor(act).type(torch.FloatTensor)
                    actions.append(act)
                    states.append(f[f'data/demo_{j}']['states'][...])

                    if self.preload:
                        obs = f[f'data/demo_{j}']['obs'][self.camera_name][...]
                        obs = obs[:,::-1, :, :]
                        obs = torch.tensor(self.preprocess_imgs(obs))
                        if self.transform is not None:
                            obs = self.transform(obs).type(torch.FloatTensor)
                        observations.append(obs)

                self.path_states.append(states)
                self.path_actions.append(actions)
                if self.preload:
                    self.path_observations.append(observations)
                # preload obs
                
        print("Finished loading data.")

        # # get action dim
        with h5py.File(self.demo_files[0], "r") as h5_file:
            demo_data = h5_file['data/demo_0']
            actions = demo_data['actions'][...]
            self.action_dim = actions.shape[-1]
        self.cumulative_sizes = list(_accumulate(self.num_paths))
        print("Num trajs per task:", self.num_paths, self.size)

        self.state_dim = self.action_dim # dummy state
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        data_file_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if data_file_idx == 0:
            path_idx = idx
        else:
            path_idx = idx - self.cumulative_sizes[data_file_idx - 1]

        act = self.path_actions[data_file_idx][path_idx]
        state = self.path_states[data_file_idx][path_idx]
        
        if self.preload:
            obs = self.path_observations[data_file_idx][path_idx]
        else:
            traj_path = Path(self.data_path) / self.processed_name / self.subtask_names[data_file_idx] / f"demo_{path_idx}"
            obs_reader = VideoReader(
                str(traj_path / "agentview_rgb.mp4"),
                num_threads=1,
            )
            obs = obs_reader.get_batch(range(len(act))) # tensor (T, h, w, c)
            obs = self.preprocess_imgs(obs)
            if self.transform is not None:
                obs = self.transform(obs)
        if torch.isnan(act).any() or torch.isnan(obs).any():
            raise ValueError(f"act/obs contains nan at idx {idx}")
        state = torch.zeros_like(act) # Note: state has different shapes across tasks. Tmp hack
        return tuple([obs, act, state, data_file_idx])

    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        data_file_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if data_file_idx == 0:
            path_idx = idx
        else:
            path_idx = idx - self.cumulative_sizes[data_file_idx - 1]

        with h5py.File(self.demo_files[data_file_idx], "r") as f:
            obs = f[f'data/demo_{path_idx}']['obs'][self.camera_name][...] # (T, h, w, c)
            return obs.shape[0]
    
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0

    @staticmethod
    def load_imgs_from_paths(paths, transform):
        imgs = []
        for path in paths:
            img = pil_loader(path)
            print(path)
            img = np.array(img).copy()
            img = torch.tensor(img.transpose(2, 0, 1) / 255)
            transformed_img = transform(img)[:3].unsqueeze(0) # (1, 3, 224, 224)
            transformed_img = transformed_img.type(torch.FloatTensor)
            imgs.append(transformed_img)
        imgs = torch.cat(imgs, dim=0)
        return imgs

def load_libero_100_slice_train_val(
    transform,
    n_rollout_per_task=50,
    data_path='/data/datasets/libero',
    split_mode="task",
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    state_based=False, # not used. No support for state_based for libero_100 yet
):
    task_name = "libero_100"
    if split_mode == "task":
        dset_train = Libero100TrajDataset(
                n_rollout_per_task=n_rollout_per_task,
                transform=transform,
                data_path=data_path,
                preload=False,
                task_name=task_name,
                tasks=list(range(90)), # use the default train/val split
            )
        
        dset_val = Libero100TrajDataset(
                n_rollout_per_task=n_rollout_per_task,
                transform=transform,
                data_path=data_path,
                preload=False,
                task_name=task_name,
                tasks=list(range(90, 100)),
            )
    else:
        raise NotImplementedError # TODO: random split

    train_slices = TrajSlicerDataset(
        dset_train, 
        num_hist + num_pred, 
        frameskip
    )
    val_slices = TrajSlicerDataset(
        dset_val, 
        num_hist + num_pred, 
        frameskip
    )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset