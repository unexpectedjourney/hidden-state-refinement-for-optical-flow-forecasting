# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from src.utils import frame_utils
from src.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, seq_len=2, overlap=True):
        self.seq_len = seq_len
        self.augmentor = None
        self.sparse = sparse

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def fix_colors(self, imgs):
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]
        return imgs

    def __getitem__(self, index):

        # if self.is_test:
        #     img1 = frame_utils.read_gen(self.image_list[index][0])
        #     img2 = frame_utils.read_gen(self.image_list[index][1])
        #     img1 = np.array(img1).astype(np.uint8)[..., :3]
        #     img2 = np.array(img2).astype(np.uint8)[..., :3]
        #     img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        #     img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        #     return img1, img2, self.extra_info[index]

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)

        flows = []
        valids = []

        for k in range(self.seq_len):
            valid = None
            flow_path = self.flow_list[index+k]
            # print(flow_path, self.sparse)
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(flow_path)
            else:
                flow = frame_utils.read_gen(flow_path)
            flows.append(flow)
            valids.append(valid)

        imgs = [frame_utils.read_gen(self.image_list[index][i]) for i in range(self.seq_len)]
        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        flows = [np.array(flow).astype(np.float32) for flow in flows]

        imgs = self.fix_colors(imgs)

        if self.augmentor is not None:
            if self.sparse:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)
            else:
                imgs, flows = self.augmentor(imgs, flows)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]

        for i in range(self.seq_len):
            if valids[i] is not None:
                valids[i] = torch.from_numpy(valids[i])
                continue
            valids[i] = (flows[i][0].abs() < 1000) & (flows[i][1].abs() < 1000)

        flows = torch.stack(flows).float()
        imgs = torch.stack(imgs).float()
        valids = torch.stack(valids).float()

        return imgs, flows, valids

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list) - self.seq_len + 1


class SeqMpiSintel(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split='training',
        root='datasets/Sintel',
        dstype='clean',
        seq_len=2,
        overlap=True,
    ):
        super(SeqMpiSintel, self).__init__(aug_params, seq_len=seq_len, overlap=overlap)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-seq_len+1):
                self.image_list += [[image_list[i+k] for k in range(seq_len)]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(
                    glob(osp.join(flow_root, scene, '*.flo'))
                )


def fetch_dataloader(args, seq_len=4, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'sintel':
        aug_params = {
            'crop_size': args.image_size,
            'min_scale': -0.2,
            'max_scale': 0.6,
            'do_flip': True
        }
        sintel_clean = SeqMpiSintel(aug_params, seq_len=seq_len, split='training', dstype='clean')
        sintel_final = SeqMpiSintel(aug_params, seq_len=seq_len, split='training', dstype='final')

        train_dataset = 100*sintel_clean + 100*sintel_final

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
