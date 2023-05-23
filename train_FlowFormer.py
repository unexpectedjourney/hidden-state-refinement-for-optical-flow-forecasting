from __future__ import print_function, division

import argparse
import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

import evaluate_FlowFormer as evaluate
import core.datasets as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger
from core.utils.flow_viz import flow_to_image

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# torch.autograd.set_detect_anomaly(True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def visualize_validation_results(model, data_blob, logger, img_name, steps, args):
    model.eval()
    with torch.no_grad():
        imgs, _, _ = data_blob
        image1 = imgs[:, 0, ...]
        image2 = imgs[:, 1, ...]
        image3 = imgs[:, 2, ...]
        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)
        image3 = image3.to(DEVICE)

        _, _, cached_data = model(image1, image2)
        cached_data["frame1"] = image1
        cached_data["frame2"] = image2
        _, _, cached_data = model(image2, image3, cached_data=cached_data)
        flow_predictions = cached_data.get("flow_predictions", [])
        flow_inertial = cached_data.get("flow_inertial")

    model.train()
    flow_predictions = [
        el.clone().detach().cpu().numpy()[0, ...] for el in flow_predictions
    ]

    flow_predictions = flow_predictions[:1] + flow_predictions[-2:]

    flow_prediction_line = np.concatenate(flow_predictions, axis=2)
    try:
        flow_prediction_line = flow_to_image(flow_prediction_line.T).T
        logger.write_img(flow_prediction_line, img_name, steps)
    except Exception as ex:
        print(ex)
    try:
        flow_inertial = flow_inertial.clone().detach().cpu().numpy()[0, ...]
        flow_inertial = flow_to_image(flow_inertial.T).T
        logger.write_img(flow_inertial, f"{img_name}_inertial", steps)
    except Exception as ex:
        print(ex)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg):
    model = nn.DataParallel(build_flowformer(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=False)
        model.module.memory_decoder.refiner.feat_encoder.load_state_dict(
            model.module.memory_encoder.feat_encoder.state_dict()
        )

    model.to(DEVICE)
    model.train()

    train_loader = datasets.fetch_dataloader(cfg, seq_len=3)
    aug_params = {
        'crop_size': cfg.image_size,
    }
    val_data = datasets.MpiSintel(
        aug_params,
        split='training',
        dstype="clean",
        seq_len=3,
    )
    val_data_blob = [el.unsqueeze(0) for el in val_data[0]]
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    add_noise = False

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            imgs, flows, valids = data_blob
            cached_data = {}

            for j in range(imgs.shape[1]-1):
                optimizer.zero_grad()
                image1 = imgs[:, j, ...]
                image2 = imgs[:, j+1, ...]
                flow = flows[:, j, ...]
                valid = valids[:, j, ...]

                image1 = image1.to(DEVICE)
                image2 = image2.to(DEVICE)
                flow = flow.to(DEVICE)
                valid = valid.to(DEVICE)

                if cfg.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (
                        image1 + stdv * torch.randn(*image1.shape).to(DEVICE)
                    ).clamp(0.0, 255.0)
                    image2 = (
                        image2 + stdv * torch.randn(*image2.shape).to(DEVICE)
                    ).clamp(0.0, 255.0)

                output = {}
                flow_predictions, cached_data = model(
                    image1, image2, output, cached_data=cached_data
                )

                cached_data["frame1"] = image1.clone().detach()
                cached_data["frame2"] = image2.clone().detach()

                if not j:
                    continue

                loss, metrics = sequence_loss(
                    flow_predictions, flow, valid, cfg
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.trainer.clip
                )

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                metrics.update(output)
                logger.push(metrics)

            # change evaluate to functions

            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                visualize_validation_results(
                    model,
                    val_data_blob,
                    logger,
                    "train-sintel",
                    total_steps,
                    args,
                )
                torch.save(model.state_dict(), PATH)

                # results = {}
                # for val_dataset in cfg.validation:
                #     if val_dataset == 'chairs':
                #         results.update(evaluate.validate_chairs(model.module))
                #     elif val_dataset == 'sintel':
                #         results.update(evaluate.validate_sintel(model.module))
                #     elif val_dataset == 'kitti':
                #         results.update(evaluate.validate_kitti(model.module))

                # logger.write_dict(results)

                model.train()

            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    PATH = f'checkpoints/{cfg.stage}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowformer',
                        help="name your experiment")
    parser.add_argument(
        '--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')

    args = parser.parse_args()

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'autoflow':
        from configs.autoflow import get_cfg
    elif args.stage == 'sintel-seq':
        from configs.sintel_seq import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(cfg)
