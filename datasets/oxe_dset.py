"""
This example shows how to use the `octo.data` dataloader with PyTorch by wrapping it in a simple PyTorch
dataloader. The config below also happens to be our exact pretraining config (except for the batch size and
shuffle buffer size, which are reduced for demonstration purposes).
"""
import numpy as np
import sys
sys.path.append("/home/kathy/dev/octo")
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import tqdm
from einops import rearrange

tf.config.set_visible_devices([], "GPU")


class TorchRLDSImageDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        transform,
        n_items=None, # TODO: check if this is #state/act pair, or #slices, or #trajs
        train=True,
        as_image=False,
        split_ratio=0.8,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self.transform = transform
        self.as_image = as_image # TODO
        self.split_ratio = split_ratio
        if n_items is None:
            self.size = self.get_len()
        else:
            self.size = min(self.get_len(), n_items)

        self.cnt = 0
        self.iterator = self._rlds_dataset.as_numpy_iterator()
        self.action_dim = 7 # TODO: hardcoded for now

        # for sample in self.iterator:
        #     processed_sample = sample['observation']['image_primary']
        #     import pdb; pdb.set_trace()
    
    def reset_state(self):
        print("Reset State...")
        self.cnt = 0
        self.iterator = self._rlds_dataset.as_numpy_iterator()
       
    def __iter__(self):
        for sample in self.iterator:
            if self.cnt < self.size:
                if self.as_image:
                    processed_sample = sample['observation']['image_primary'].squeeze(0)
                else:
                    processed_sample = sample['observation']['image_primary']
                processed_sample = torch.tensor(processed_sample)
                processed_sample = self.preprocess_imgs(processed_sample)
                processed_sample = self.transform(processed_sample)
                self.cnt += 1
                # TODO: dummy state
                yield processed_sample, sample['action'], torch.zeros_like(torch.tensor(sample['action']))# only use primary camera image for now, after squeeze: [b, h]
            else:
                break
            
    def preprocess_imgs(self, imgs):
        """
        Reshape imgs from to (T, h, w, c) to (T, c, h, w)
        """
        imgs = rearrange(imgs, "T H W C-> T C H W") / 255.
        return imgs

    def get_len(self):
        lengths = np.array(
            [
                stats["num_transitions"].astype('float64')
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"): # sample_weights addup to 1, lengths might not be an integer
            lengths *= np.array(self._rlds_dataset.sample_weights)
        lengths = np.floor(lengths).astype('int64')
        total_len = lengths.sum()
        if self._is_train:
            return int(self.split_ratio * total_len)
        else:
            return int(self.split_ratio * total_len)

    def __len__(self):
        return self.size
        
def make_oxe_dset(
        transform, 
        data_path, 
        name='oxe_magic_soup', 
        is_train=True, 
        window_size=1,
        n_items=None,
        as_image=False,
        split_ratio=0.8,
    ):

    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        name,
        data_path,
        load_camera_views=("primary", "wrist"),
    )
    print("interleaving...")
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=is_train,
        shuffle_buffer_size=1,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
        batch_size=None,  # batching will be handles in PyTorch Dataloader object
        balance_weights=True,
        traj_transform_kwargs=dict(
            goal_relabeling_strategy="uniform",
            window_size=window_size,
            future_action_window_size=0, # ?
            subsample_length=100,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
                "wrist": dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            },
            resize_size=dict(
                primary=(256, 256),
                wrist=(128, 128),
            ),
            num_parallel_calls=10,
        ),
        traj_transform_threads=10,
        traj_read_threads=10,
    )

    pytorch_dataset = TorchRLDSImageDataset(
        dataset, 
        transform=transform,
        n_items=n_items,
        train=is_train,
        as_image=as_image,
        split_ratio=split_ratio
    )
    return pytorch_dataset


def load_oxe_slice_train_val(
    name,
    transform,
    data_path,
    n_items=1,
    num_hist=0,
    num_pred=0,
    split_ratio=0.8,
    frameskip=1, # TODO: add support for this, currently can only be 1
    state_based=False, # not used. No support for state_based for oxe yet
):
    train_dset = make_oxe_dset(
        transform,
        data_path,
        name=name,
        is_train=True,
        window_size=num_hist + num_pred,
        n_items=n_items,
        as_image=False,
        split_ratio=split_ratio
    )
    
    val_dset = make_oxe_dset(
        transform,
        data_path,
        name=name,
        is_train=False,
        window_size=num_hist + num_pred,
        n_items=n_items,
        as_image=False,
        split_ratio=split_ratio
    )

    datasets = {}
    datasets['train'] = train_dset
    datasets['valid'] = val_dset
    return datasets, {'train': [], 'valid': []} # No traj dataset for oxe (TODO: add support?)

