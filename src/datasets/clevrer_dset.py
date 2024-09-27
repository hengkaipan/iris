import os
import cv2
import numpy as np
import yaml
import torch
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from hydra import compose, initialize
from omegaconf import OmegaConf
from .CLEVRER.temporal_reasoning.data import VisualCLEVRDataset
from src.utils_iris import pil_loader
# from utils import dict_to_namespace

class CLEVRERWrapperDataset(Dataset):
    def __init__(self, frameskip, split_ratio, num_hist, num_pred, n_rollout, transform, phase):
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(base_path, 'conf/env/clevrer_default.yaml')) as f:
            clevrer_args = yaml.safe_load(f)
        clevrer_args['frame_offset'] = frameskip
        clevrer_args['n_rollout'] = n_rollout
        clevrer_args['n_his'] = num_hist + num_pred - 2
        clevrer_args['data_dir'] = os.path.join(base_path, clevrer_args['data_dir'])
        clevrer_args['label_dir'] = os.path.join(base_path, clevrer_args['label_dir'])
        clevrer_args['train_valid_ratio'] = split_ratio
        clevrer_args = Namespace(**clevrer_args)
        # clevrer_args = dict_to_namespace(clevrer_args) # TODO: preference for this or Namespace(**clevrer_args)?
        self.clevrer_dset = VisualCLEVRDataset(clevrer_args, transform, phase)
        self.action_dim = 1 # dummy action
    
    def __len__(self):
        return self.clevrer_dset.__len__()
    
    def __getitem__(self, idx):
        item = self.clevrer_dset.__getitem__(idx)
        return item, torch.zeros([item.shape[0], 1]), torch.zeros([item.shape[0], 1]) # dummy action, state
    
    @staticmethod
    def load_imgs_from_paths(paths, transform):
        imgs = []
        for path in paths:
            img = pil_loader(path)
            print(path)
            img = np.array(img)[:, :, ::-1].copy()
            img = cv2.resize(img, (150, 100), interpolation=cv2.INTER_AREA).astype(float) / 255. # original img shape: (320, 480, 3)
            transformed_img = transform(img)[:3].unsqueeze(0) # (1, 3, 224, 224)
            transformed_img = transformed_img.type(torch.FloatTensor)
            imgs.append(transformed_img)
        imgs = torch.cat(imgs, dim=0)
        return imgs


def load_clevrer_slice_train_val(
    transform,
    n_rollout=50,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    state_based=False, # doesn't apply for clevrer
):
    datasets = {phase: CLEVRERWrapperDataset(
        split_ratio=split_ratio,
        frameskip=frameskip,
        num_hist=num_hist,
        num_pred=num_pred,
        n_rollout=n_rollout,
        transform=transform,
        phase=phase
    )  for phase in ['train', 'valid']}

    return datasets, {'train': [], 'valid': []} # No traj dataset for CLEVRER
    
if __name__ == "__main__":
    from torchvision import datasets, transforms, utils
    img_size = 224
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)
    dset = CLEVRERWrapperDataset(
        frameskip=20,
        num_hist=5,
        num_pred=1,
        n_rollout=20,
        transform=transform,
        phase='train'
    )

    dataloader = torch.utils.data.DataLoader(
        dset, batch_size=7,
        shuffle=True, 
        num_workers=1,
        collate_fn=None)
    
    for i, data in enumerate(dataloader):
        # data: [b, num_hist + num_pred, 3, img_size, img_size]
        import pdb; pdb.set_trace()