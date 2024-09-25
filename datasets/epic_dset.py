import os
import cv2
from torch.utils.data import Dataset
import json 
import bisect
import torch 
from torch.utils.data import DataLoader, IterableDataset
import random
import time
from tqdm import tqdm
import einops
import decord
from decord import VideoReader, VideoLoader
from decord import cpu, gpu
decord.bridge.set_bridge("torch")

class EpicKitchenVideoDataset(Dataset):
    def __init__(
        self,
        folder='/vast/gz2123/datasets/torrent/3h91syskeag572hl6tvuovwv4d/videos',
        transform=None,
        img_size=[270, 480],
        frameskip=1,
        window_size=1,
        mode="train",
        metadata="metadata.json",
        as_image=False,
        n_videos=None,
    ) -> None:
        
        self.folder = folder
        self.frameskip = frameskip
        self.window_size = window_size
        self.transform = transform
        self.as_image = as_image
        if not os.path.exists(os.path.join(folder, "metadata.json")):
            self.gen_metadata()
        with open(os.path.join(self.folder, "metadata.json")) as json_file:
            metadata = json.load(json_file)
        self.metadata = metadata

        num_frames = 0
        num_videos = 0
        self.video_paths = []
        cumulative_sizes = []
        for key in sorted(metadata.keys()):
            if mode in key:
                num_videos += 1
                num_frames += metadata[key]
                cumulative_sizes.append(num_frames)
                self.video_paths.append(os.path.join(self.folder, key))

                if n_videos is not None and num_videos >= n_videos:
                    break
        # self.video_paths = [self.video_paths[1]]
        cur_time = time.time()
        print("Create loader begin...")
        video_loader = VideoLoader(
            self.video_paths,
            ctx=cpu(0),
            shape=[window_size] + img_size + [3],
            interval=1, 
            skip=frameskip, 
            shuffle=0
        )
        print("Finished creating loader, time used: ", time.time() - cur_time)
        self.size = len(video_loader)
        self.video_loader = video_loader
        print("Dataset size: ", self.size)
        self.dummy = None
        self.error_count = 0

        self.action_dim = 1 # dummy action
    
    def __len__(self):
        return self.size

    def gen_metadata(self):
        video_metadata = {}
        folder_path = self.folder
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder) # train/test
            for videofolder in os.listdir(subfolder_path):
                videofolder_path = os.path.join(subfolder_path, videofolder)
                print(videofolder_path)
                for video_file in os.listdir(videofolder_path):
                    print(video_file)
                    video_path = os.path.join(videofolder_path, video_file)
                    reader = VideoReader(
                        video_path,
                        num_threads=1,
                    )
                    frames_count = len(reader)
                    video_metadata[f'{subfolder}/{videofolder}/{video_file}'] = frames_count

        with open(os.path.join(self.folder, "metadata.json"), 'w') as json_file:
            json.dump(video_metadata, json_file, indent=4)

    def reset_state(self):
        print("Prev epoch error count: ", self.error_count)
        self.error_count = 0
        self.video_loader.reset()
        
    def __getitem__(self, idx): 
        # print(f"### get item {idx}")
        while True:
            try: 
                frames = self.video_loader.next()[0]
                frames = self.preprocess_imgs(frames)
                if self.transform: # apply transform after loading batch for speed 
                    frames = self.transform(frames)
                if self.as_image:
                    frames = frames[0]
                if self.dummy is None: # record a dummy item
                    self.dummy = frames
                return frames, torch.zeros([frames.shape[0], 1]), torch.zeros([frames.shape[0], 1]) # dummy action, state
            except StopIteration:
                self.reset_state() # this should not be called
            except Exception as e:
                # import pdb; pdb.set_trace()
                self.error_count += 1
                return self.dummy, torch.zeros([self.dummy.shape[0], 1]), torch.zeros([self.dummy.shape[0], 1])  # return dummy element if there is an error, assume dummy is not None
    
    def __len__(self):
        return self.size
    
    def preprocess_imgs(self, imgs):
        """
        Reshape imgs from to (T, h, w, c) to (T, c, h, w)
        """
        imgs = einops.rearrange(imgs, "T H W C-> T C H W") / 255.
        return imgs
    
def load_epic_slice_train_val(
    transform,
    data_path,
    n_videos=1,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    state_based=False, # doesn't apply for epic
):
    train_dset = EpicKitchenVideoDataset(
            folder=data_path,
            transform=transform,
            frameskip=frameskip, 
            n_videos=n_videos,
            window_size=num_hist + num_pred, 
            mode="train",
            as_image=False
        )
    
    val_dset = EpicKitchenVideoDataset(
            folder=data_path,
            transform=transform,
            frameskip=frameskip, 
            n_videos=n_videos,
            window_size=num_hist + num_pred, 
            mode="test",
            as_image=False
        )
    
    datasets = {}
    datasets['train'] = train_dset
    datasets['valid'] = val_dset
    return datasets,  {'train': [], 'valid': []} # No traj dataset for epic



if __name__ == "__main__":
    from torchvision import datasets, transforms, utils
    transform = transforms.Compose(
        [   
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # not normalize at dataloader for memory consideration
        ]
    )
    transform2 = transforms.Compose(
        [   
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # not normalize at dataloader for memory consideration
        ]
    )
    # dset = EpicKitchenVideoDatasetV2(
    #     folder="/home/kathy/Downloads/videos", 
    #     # img_size=224, 
    #     transform=transform2, 
    #     frameskip=5, 
    #     window_size=7, 
    #     mode="train"
    # )

    dset = EpicKitchenVideoDataset(folder="/home/kathy/Downloads/videos", transform=transform, frameskip=5, window_size=7, mode="train")
    dataloader = DataLoader(dset, batch_size=128, shuffle=False, num_workers=0)
    # time the dataloader
    import time
    cur_time = time.time()
    for i, (data, _) in enumerate(dataloader):
        b = data.shape[0]
        data = transform2(einops.rearrange(data.float(), "B T C H W -> (B T) C H W"))
        data = einops.rearrange(data, "(B T) C H W -> B T C H W", B=b)
        print(data.shape, time.time() - cur_time)

        import pdb; pdb.set_trace()
