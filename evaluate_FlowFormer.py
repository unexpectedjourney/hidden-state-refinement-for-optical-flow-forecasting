import torch
import numpy as np
import os
import argparse

import core.datasets as datasets
from core.utils.utils import InputPadder
from core.FlowFormer import build_flowformer
from core.utils import frame_utils
from configs.small_things_eval import get_cfg as get_small_things_cfg
from configs.things_eval import get_cfg as get_things_cfg

# from FlowFormer import FlowFormer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def validate_chairs(model):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        imgs, flow_gts, _ = val_dataset[val_id]
        image1 = imgs[0, ...]
        image2 = imgs[1, ...]
        image1 = image1[None].to(DEVICE)
        image2 = image2[None].to(DEVICE)
        flow_pre, _ = model(image1, image2)

        epe = torch.sum((flow_pre[0].cpu() - flow_gts[0])**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            imgs, flow_gts, _ = val_dataset[val_id]
            image1 = imgs[0, ...]
            image2 = imgs[1, ...]
            image1 = image1[None].to(DEVICE)
            image2 = image2[None].to(DEVICE)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pre = model(image1, image2)

            flow_pre = padder.unpad(flow_pre[0]).cpu()[0]

            epe = torch.sum((flow_pre - flow_gts[0])**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" %
              (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def create_sintel_submission(model, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """

    model.eval()
    for dstype in ['final', "clean"]:
        test_dataset = datasets.MpiSintel(
            split='test', aug_params=None, dstype=dstype)

        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
            imgs, (sequence, frame) = test_dataset[test_id]

            for j in range(imgs.shape[0] - 1):
                image1 = imgs[j, ...]
                image2 = imgs[j+1, ...]

                image1 = image1[None].to(DEVICE)
                image2 = image2[None].to(DEVICE)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_pre = model(image1, image2)

                flow_pre = padder.unpad(flow_pre[0]).cpu()
                flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

                output_dir = os.path.join(output_path, dstype, sequence)
                output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                frame_utils.writeFlow(output_file, flow)


@torch.no_grad()
def validate_kitti(model):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        imgs, flow_gts, valid_gts = val_dataset[val_id]
        image1 = imgs[0, ...]
        image2 = imgs[1, ...]
        flow_gt = flow_gts[0]
        valid_gt = valid_gts[0]

        image1 = image1[None].to(DEVICE)
        image2 = image2[None].to(DEVICE)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre = model(image1, image2)

        flow_pre = padder.unpad(flow_pre[0]).cpu()[0]

        epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args()
    # cfg = get_cfg()
    if args.small:
        cfg = get_small_things_cfg()
    else:
        cfg = get_things_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    print(args)

    model.to(DEVICE)
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'sintel_submission':
            create_sintel_submission(model.module)