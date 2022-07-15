import os

import cv2
import numpy as np

from lib.config import args, cfg


def run_dataset():
    import tqdm

    from lib.datasets import make_data_loader

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    import time

    import torch
    import tqdm

    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != "meta":
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch["inp"])
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    import torch
    import tqdm

    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    if cfg.rotate_reproduce:
        cfg.segm_or_bbox = "segm"
        data_loader = make_data_loader(cfg, is_train=False)
        # rot_data_loader = make_data_loader(cfg, is_train=False)
        evaluator = make_evaluator(cfg)

        for batch in tqdm.tqdm(data_loader):
            inp = batch["inp"].cuda()
            # inp_rotated = torch.from_numpy(cv2.rotate(batch['inp'].numpy().reshape(512, 512, 3), cv2.ROTATE_90_CLOCKWISE).reshape(1, 3, 512, 512)).cuda()
            with torch.no_grad():
                output = network(inp, batch)
                # output_rot = network(inp_rotated, batch)

            evaluator.evaluate_rotate(output, batch)
            # evaluator.evaluate_rotate(output_rot, batch, True)

        cfg.rotate = 90
        for batch in tqdm.tqdm(data_loader):
            inp = batch["inp"].cuda()
            with torch.no_grad():
                output = network(inp, batch)
            evaluator.evaluate_rotate(output, batch, rotate=True)

        evaluator.summarize_rotate()

    else:
        data_loader = make_data_loader(cfg, is_train=False)
        evaluator = make_evaluator(cfg)
        for batch in tqdm.tqdm(data_loader):
            inp = batch["inp"].cuda()
            with torch.no_grad():
                output = network(inp, batch)
            evaluator.evaluate(output, batch)
        evaluator.summarize()


def run_visualize():
    import torch
    import tqdm

    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != "meta":
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch["inp"], batch)
        visualizer.visualize(output, batch)


def run_sbd():
    from tools import convert_sbd

    convert_sbd.convert_sbd()


def run_demo():
    from tools import demo

    demo.demo()


if __name__ == "__main__":
    globals()["run_" + args.type]()
