"""
PyTorch dataset for loading snapshots of ModelNet 3D meshes.
"""
import os
import random
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import random

import torchvision.transforms as T
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

classes = {
  "bathtub": 0,
  "bed": 1,
  "chair": 2,
  "desk": 3,
  "dresser": 4,
  "monitor": 5,
  "night_stand": 6,
  "sofa": 7,
  "table": 8,
  "toilet": 9
}

class ModelNetSnapshot(torch.utils.data.Dataset):
    """
    Dataset for loading ModelNet 3D mesh snapshots.
    """
    dataset_path = ""
    classes_cnts = {}
    
    def __init__(self, split, dataset_path='ModelNet10/', augment=True) -> None:
        super().__init__()
        assert split in ['train', 'test', 'validation']
        
        classes_cnt = {
            "bathtub": 0,
            "bed": 0,
            "chair": 0,
            "desk": 0,
            "dresser": 0,
            "monitor": 0,
            "night_stand": 0,
            "sofa": 0,
            "table": 0,
            "toilet": 0
        }
        
        classes_snaps = {
            "bathtub": [],
            "bed": [],
            "chair": [],
            "desk": [],
            "dresser": [],
            "monitor": [],
            "night_stand": [],
            "sofa": [],
            "table": [],
            "toilet": []
        }
        
        self.snapshots = []
        self.dataset_path = dataset_path
        self.augment=augment
        
        for root, dirs, files in os.walk(dataset_path):
            for dir in dirs:
                if 'snapshots' in dir and split in root:
                    if split == "train":
                        classes_cnt[root.split("/")[-2]] += 1
                        classes_snaps[root.split("/")[-2]].append(os.path.join(root, dir))
                        
                    self.snapshots.append(os.path.join(root, dir))
         
        self.rand_trans = torch.nn.ModuleList([
            torch.nn.Sequential(T.CenterCrop(80), T.Resize(256)),
            torch.nn.Sequential(T.CenterCrop(150), T.Resize(256)),
            torch.nn.Sequential(T.CenterCrop(200), T.Resize(256)),
            T.RandomRotation((90, 90)), 
            T.RandomRotation((180, 180)), 
            T.RandomRotation((270, 270)), 
        ])
        self.rand_trans_p =  0.3
        
        if split == "train":
            for cat in classes_snaps:
                rand_pick = random.choices(classes_snaps[cat], k=(classes_cnt[cat] // 3))
                appendable = [snapshot + "_transformed" for snapshot in rand_pick]
                self.snapshots += appendable

    def __getitem__(self, index):
        
        snapshot_path = self.snapshots[index]
        
        # Construct the filenames
        view_patches = []
        transform = random.choice(self.rand_trans) if ((np.random.uniform() < self.rand_trans_p) or ("_transformed" in snapshot_path)) else None
        snapshot_path = snapshot_path if ("_transformed" not in snapshot_path) else snapshot_path.replace("_transformed", "")
        
        for i in range(12):
            fn = os.path.join(snapshot_path, f'{str(i)}.png')
            if self.augment:
                view_patches.append(self.split_image(fn, transform=transform))
            else:
                view_patches.append(self.split_image(fn))

        
        view_patches = np.array(view_patches)
            
        # Construct the patches for each view
        label = self.snapshots[index].split('/')[-3]
        
        view_patches = torch.tensor(view_patches, dtype=torch.float32) / 255
        
        return {
            # potentionally slow
            'view_patches': view_patches,
            'labels': classes[label]
        }
    
    def __len__(self):
        return len(self.snapshots)
    
    def split_image(self, img_path, patch_size=4, transform=None):
        im = Image.open(img_path)
        im = np.asarray(transform(im)) if transform else np.asarray(im)
        assert im.shape[0] == im.shape[1]
        
        M = N = im.shape[0] // patch_size

        # Source: https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
        tiles = np.array([im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)])

        return tiles

## TODO: adapt code for our purposes
class MODELNETDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ModelNetSnapshot(dataset_path=self.hparams.data_dir, split="train")
            self.data_test = ModelNetSnapshot(dataset_path=self.hparams.data_dir, split="test", augment=False)
            self.data_val = ModelNetSnapshot(dataset_path=self.hparams.data_dir, split="validation", augment=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )


if __name__ == '__main__':
    ds = ModelNetSnapshot('train')
    print(ds[0]['view_patches'].shape)
        
        
        
        