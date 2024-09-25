import os
import h5py
import torch
import bisect
from tqdm import tqdm
import numpy as np
from typing import Callable, Optional, List
from .traj_dset import _accumulate, TrajSlicerDataset, TrajDataset, get_train_val_sliced
from libero.libero import benchmark
from utils import pil_loader
from einops import rearrange


NUM_PATHS_PER_TASK = 50
# stats for the entire dataset
ACTION_MEAN = torch.tensor([0.0417,  0.0320, -0.1499, -0.0019,  0.0248,  0.0262, -0.2942])
ACTION_STD = torch.tensor([0.3980, 0.3535, 0.4873, 0.0563, 0.0782, 0.0989, 0.9558])

STATE_MEAN = torch.tensor([ 3.7576e+00,  4.9865e-02,  4.6627e-01,  2.5652e-02, -1.8998e+00,
        -2.3258e-01,  2.1918e+00,  7.0258e-01,  2.8381e-02, -2.7264e-02,
        -8.6240e-02, -2.4333e-03,  9.2182e-01,  7.1891e-01,  7.7180e-03,
        -7.4080e-03,  6.8779e-01, -5.1751e-02,  1.2554e-01,  9.1198e-01,
         6.2246e-04,  2.7274e-03, -1.5302e-03,  9.9819e-01, -1.9259e-01,
        -6.0033e-02,  9.2369e-01, -7.6372e-04,  1.1007e-04, -2.2793e-02,
         9.9158e-01,  5.0581e-02, -8.3118e-03,  9.0265e-01,  7.0579e-01,
        -1.8580e-03,  1.3345e-03,  7.0804e-01, -1.8821e-02, -2.6019e-03,
         4.4323e-06,  8.9536e-03,  6.8436e-03,  1.2344e-01,  3.8074e-03,
         1.2816e-01, -3.1524e-02, -3.5349e-02, -2.6450e-02, -1.6623e-04,
        -6.9265e-05,  2.9337e-03, -1.9357e-03,  6.7922e-03, -2.5500e-03,
         5.0395e-04, -1.6883e-02, -9.8118e-04, -2.2011e-03,  4.8009e-04,
        -1.0632e-03, -4.5155e-03, -7.4332e-04,  3.0579e-03, -4.3811e-03,
         7.6656e-03, -1.6387e-02,  1.3570e-03,  2.7596e-03, -1.5806e-03,
         3.4332e-03,  6.7519e-05,  1.1419e-03,  1.3635e-03, -1.0974e-03,
        -2.3785e-03, -2.4482e-03,  9.6173e-07,  1.3383e-02])
STATE_STD = torch.tensor([2.4737e+00, 1.0381e-01, 3.5946e-01, 1.4763e-01, 5.0269e-01, 3.9882e-01,
        4.0187e-01, 5.6051e-01, 1.4849e-02, 1.4914e-02, 3.7816e-02, 4.6071e-02,
        6.5829e-02, 4.9098e-02, 3.7826e-02, 3.5945e-02, 6.9705e-02, 1.3289e-02,
        2.3276e-02, 1.7264e-02, 1.5894e-02, 3.5860e-02, 4.1559e-02, 1.8653e-02,
        3.3739e-02, 3.4851e-02, 7.4385e-02, 4.0683e-02, 6.2828e-02, 9.4417e-02,
        4.1577e-02, 1.3938e-02, 4.6333e-02, 9.5457e-04, 1.2088e-02, 1.0952e-02,
        1.0007e-02, 1.2437e-02, 5.0410e-02, 1.7978e-02, 7.9378e-05, 7.1235e-02,
        8.5447e-02, 3.1457e-01, 1.1417e-01, 4.0666e-01, 2.1504e-01, 3.5417e-01,
        2.7228e-01, 2.5457e-02, 2.5897e-02, 3.7219e-02, 3.8095e-02, 6.3950e-02,
        2.7423e-01, 4.1414e-01, 2.9907e-01, 1.0357e-02, 1.9246e-02, 3.0525e-02,
        2.5019e-01, 2.7203e-01, 7.4880e-02, 2.2160e-02, 2.1474e-02, 4.3316e-02,
        1.6080e-01, 1.7027e-01, 1.6728e-01, 1.4366e-02, 1.8969e-02, 9.3687e-03,
        1.9177e-01, 2.3204e-01, 4.8083e-02, 1.8762e-02, 1.8375e-02, 2.1868e-04,
        9.1398e-02])

PROPRIO_MEAN = torch.tensor([ 0.0284, -0.0273, -0.0925,  0.0121,  1.0687,  0.9168,  0.1214, -0.1000,
         0.0679])
PROPRIO_STD = torch.tensor([0.0149, 0.0149, 0.1145, 0.1156, 0.1046, 0.1508, 0.2695, 0.1359, 0.1274])

class LiberoTrajDataset(TrajDataset):
    def __init__(
            self, 
            n_rollout_per_task: Optional[int] = 50,
            transform: Optional[Callable] = None,
            data_path: str = '/home/kathy/dev/LIBERO/libero/datasets/',
            preload: bool = True,
            normalize_action: bool = False,
            task_name: str = 'libero_goal', # use libero_goal as default
            tasks: Optional[List[int]] = None, # use all tasks if not specifying tasks
            state_based: bool = False,
        ):
        self.num_paths = []
        self.camera_name = 'agentview_rgb'
        self.transform = transform
        self.preload = preload
        self.normalize_action = normalize_action
        self.path_observations = []
        self.path_actions = []
        self.path_states = []
        self.path_proprios = []
        self.state_based = state_based
        benchmark_dict = benchmark.get_benchmark_dict()

        if task_name == 'libero_100':
            benchmark_instance_90 = benchmark_dict['libero_90']()
            num_tasks_90 = benchmark_instance_90.get_num_tasks()
            demo_files_90 = [benchmark_instance_90.get_task_demonstration(i) for i in range(num_tasks_90)]
            demo_files_90 = [
                os.path.join(data_path, file) for file in demo_files_90
            ]
            
            benchmark_instance_10 = benchmark_dict['libero_10']()
            num_tasks_10 = benchmark_instance_10.get_num_tasks()
            demo_files_10 = [benchmark_instance_10.get_task_demonstration(i) for i in range(num_tasks_10)]
            demo_files_10 = [
                os.path.join(data_path, file) for file in demo_files_10
            ]
            self.demo_files = demo_files_90 + demo_files_10
            num_tasks = num_tasks_90 + num_tasks_10
        else:
            benchmark_instance = benchmark_dict[task_name]()
            num_tasks = benchmark_instance.get_num_tasks()
            demo_files = [benchmark_instance.get_task_demonstration(i) for i in range(num_tasks)]
            self.demo_files = [
                os.path.join(data_path, file) for file in demo_files
            ]


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
        print("Num trajs per task:", self.num_paths, self.size)

        # get dims
        with h5py.File(self.demo_files[0], "r") as h5_file:
            demo_data = h5_file['data/demo_0']
            self.action_dim = demo_data['actions'][...].shape[-1]
            self.state_dim = demo_data['states'][...].shape[-1]
            self.proprio_dim = demo_data['robot_states'][...].shape[-1]
        self.cumulative_sizes = list(_accumulate(self.num_paths))

        if self.normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN
            self.state_std = STATE_STD
            self.proprio_mean = PROPRIO_MEAN
            self.proprio_std = PROPRIO_STD
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        if self.preload:
            for t in tqdm(range(num_tasks), desc="Loading Data"):
                demo_file = self.demo_files[t]
                with h5py.File(demo_file, "r") as f:
                    observations = []
                    actions = []
                    states = []
                    proprios = []
                    for j in range(self.num_paths[t]): # load the first n trajs
                        if not self.state_based:
                        # if False:
                            obs = f[f'data/demo_{j}']['obs'][self.camera_name][...]
                            obs = obs[:,::-1, :, :]
                            obs = torch.tensor(self.preprocess_imgs(obs))
                            if self.transform is not None:
                                obs = self.transform(obs).type(torch.FloatTensor)
                            observations.append(obs)
                        else:
                            observations.append(None)
                        act = torch.tensor(f[f'data/demo_{j}']['actions'][...]).type(torch.FloatTensor)
                        state = torch.tensor(f[f'data/demo_{j}']['states'][...]).type(torch.FloatTensor)
                        proprio = torch.tensor(f[f'data/demo_{j}']['robot_states'][...]).type(torch.FloatTensor)

                        # only normalize state when it's used as obs
                        act = (act - self.action_mean) / self.action_std
                        proprio = (proprio - self.proprio_mean) / self.proprio_std

                        actions.append(act)
                        states.append(state)
                        proprios.append(proprio)
                    self.path_observations.append(observations)
                    self.path_actions.append(actions)
                    self.path_states.append(states)
                    self.path_proprios.append(proprios)
            print("Finished loading data.")

    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # print(f"Libero getting item {idx}")
        data_file_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if data_file_idx == 0:
            path_idx = idx
        else:
            path_idx = idx - self.cumulative_sizes[data_file_idx - 1]

        # print(f"{self.num_paths} Libero getting item {idx}, data_file_idx: {data_file_idx}, path_idx: {path_idx}")
        if self.preload:
            act = self.path_actions[data_file_idx][path_idx]
            state = self.path_states[data_file_idx][path_idx]
            proprio = self.path_proprios[data_file_idx][path_idx]
            if not self.state_based:
                visual = self.path_observations[data_file_idx][path_idx]
            else:
                visual = (state - self.state_mean) / self.state_std
                visual = visual.type(torch.FloatTensor)
            # act = torch.ones_like(act)*data_file_idx # TODO: change it back after debugging
            if torch.isnan(act).any() or torch.isnan(visual).any():
                raise ValueError(f"act/obs contains nan at idx {idx}")
            obs = {
                "visual": visual,
                "proprio": proprio
            }
            return tuple([obs, act, state, data_file_idx])
        else:
            with h5py.File(self.demo_files[data_file_idx], "r") as f:
                act = torch.tensor(f[f'data/demo_{path_idx}']['actions'][...]).type(torch.FloatTensor)
                state = torch.tensor(f[f'data/demo_{path_idx}']['states'][...]).type(torch.FloatTensor)
                proprio = torch.tensor(f[f'data/demo_{path_idx}']['robot_states'][...]).type(torch.FloatTensor)
                
                # only normalize state when it's used as obs
                act = (act - self.action_mean) / self.action_std
                proprio = (proprio - self.proprio_mean) / self.proprio_std
                
                if not self.state_based:
                    visual = f[f'data/demo_{path_idx}']['obs'][self.camera_name][...]
                    visual = visual[:,::-1, :, :] # Libero dataset images are reversed for some reason
                    visual = torch.tensor(self.preprocess_imgs(visual))
                    if self.transform is not None:
                        visual = self.transform(visual).type(torch.FloatTensor)
                else:
                    visual = (state - self.state_mean) / self.state_std
                    visual = visual.type(torch.FloatTensor)
                obs = {
                    "visual": visual,
                    "proprio": proprio
                }
                if torch.isnan(act).any() or torch.isnan(visual).any():
                    raise ValueError(f"act/obs contains nan at idx {idx}")
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
        """
        Reshape imgs from to (T, h, w, c) to (T, c, h, w)
        """
        # if imgs type is np array
        if isinstance(imgs, np.ndarray):
            imgs = imgs.transpose(0, 3, 1, 2) / 255.
        elif isinstance(imgs, torch.Tensor):
            imgs = rearrange(imgs, 'b h w c -> b c h w') / 255.
        # imgs = imgs.transpose(0, 3, 1, 2) / 255.
        return imgs
    
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
    

def load_libero_goal_slice_train_val(
    transform,
    n_rollout_per_task=50,
    data_path='/data/datasets/libero',
    split_mode="task",
    normalize_action=True,
    train_tasks=[0, 1, 2, 3, 4, 5, 6, 7],
    val_tasks=[8, 9],
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    state_based=False,
):
    task_name = "libero_goal"
    if split_mode == "task":
        dset_train = LiberoTrajDataset(
                n_rollout_per_task=n_rollout_per_task,
                transform=transform,
                data_path=data_path,
                preload=True,
                normalize_action=normalize_action,
                task_name=task_name,
                tasks=train_tasks,
                state_based=state_based,
            )
        
        dset_val = LiberoTrajDataset(
                n_rollout_per_task=n_rollout_per_task,
                transform=transform,
                data_path=data_path,
                preload=True,
                normalize_action=normalize_action,
                task_name=task_name,
                tasks=val_tasks,
                state_based=state_based,
            )

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
    elif split_mode == "random":
        dset = LiberoTrajDataset(
                n_rollout_per_task=n_rollout_per_task,
                transform=transform,
                data_path=data_path,
                preload=True,
                normalize_action=normalize_action,
                task_name=task_name,
                tasks=train_tasks,
                state_based=state_based,
            )
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset, 
            train_fraction=split_ratio, 
            num_frames=num_hist + num_pred, 
            frameskip=frameskip
        )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset


if __name__ == "__main__":
    from torchvision import datasets, transforms, utils
    img_size = 224
    transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)
    dset = LiberoTrajDataset(
        n_rollout=50,
        transform=transform,
    )

    train_dset, val_dset = get_train_val_sliced(dset, train_fraction=0.5, num_frames=4)

    dataloader = torch.utils.data.DataLoader(
        train_dset, batch_size=7,
        shuffle=True, 
        num_workers=1,
        collate_fn=None)
    
    for i, data in enumerate(dataloader):
        # data: [b, num_hist + num_pred, 3, img_size, img_size]
        import pdb; pdb.set_trace()