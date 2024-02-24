import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import re

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
    cfg.use_val = False

    # get ap for each model
    # epoch = 29
    # while epoch < 30:
    
    load_network(network, cfg.model_dir, epoch=best_model)
    print('--------------------------testing epoch = {}----------------------------------'.format(best_model))
    network.eval()

    if cfg.rotate_reproduce:
        cfg.segm_or_bbox = "segm"
        data_loader = make_data_loader(cfg, is_train=False)
        # rot_data_loader = make_data_loader(cfg, is_train=False)
        evaluator = make_evaluator(cfg)

        for batch in tqdm.tqdm(data_loader):
            for key, value in batch.items():
                if key == 'meta':
                    continue
                batch[key] = batch[key].cuda()
            # inp = batch["inp"].cuda()
            # inp_rotated = batch["inp"].cuda()
            # print(inp_rotated.shape)
            inp = batch["inp"].cuda()
            inp_rotated = torch.rot90(inp, k=-1, dims=[2, 3])

            # inp_rotated = torch.from_numpy(cv2.rotate(batch['inp'].numpy().reshape(512, 512, 3), cv2.ROTATE_90_CLOCKWISE).reshape(1, 3, 512, 512)).cuda()
            with torch.no_grad():
                # output = network(inp, batch)
                output_rot = network(inp_rotated, batch)

            # evaluator.evaluate_rotate(output, batch)
            evaluator.evaluate_rotate(output_rot, batch, True)

        # cfg.rotate = 90
        # for batch in tqdm.tqdm(data_loader):           
        #     for key, value in batch.items():
        #         if key == 'meta':
        #             continue
        #         batch[key] = batch[key].cuda()
        #     inp = batch["inp"].cuda()
        #     with torch.no_grad():
        #         output = network(inp, batch)
        #     evaluator.evaluate_rotate(output, batch, rotate=True)

        evaluator.summarize_rotate()

    else:
        data_loader = make_data_loader(cfg, is_train=False)

        if cfg.segm_or_bbox == "both":
            cfg.segm_or_bbox = "segm"
            segm_evaluator = make_evaluator(cfg)
            cfg.segm_or_bbox = "bbox"
            det_evaluator = make_evaluator(cfg)
            cfg.segm_or_bbox = "both"
            for batch in tqdm.tqdm(data_loader):
                for key, value in batch.items():
                    if key == 'meta':
                        continue
                    batch[key] = batch[key].cuda()
                inp = batch["inp"]
                with torch.no_grad():
                    output = network(inp, batch)
                det_evaluator.evaluate(output, batch)
                segm_evaluator.evaluate(output, batch)
            det_evaluator.summarize()
            segm_evaluator.summarize()
        else:
            evaluator = make_evaluator(cfg)
            for batch in tqdm.tqdm(data_loader):
                for key, value in batch.items():
                    if key == 'meta':
                        continue
                    batch[key] = batch[key].cuda()
                inp = batch["inp"]
                with torch.no_grad():
                    output = network(inp, batch)
                evaluator.evaluate(output, batch)
            evaluator.summarize()
    # epoch += 1

def run_visualize():
    import torch
    import tqdm

    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from lib.visualizers import make_visualizer
    cfg.use_val = False

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=best_model)
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


def find_highest_segm_ap(log_file_path):
    max_ap = -1
    max_step = -1
    
    with open(log_file_path, 'r') as f:
        for line in f:
            if "val/segm_ap - Step" in line:
                match = re.search(r"Step: (\d+), Value: ([0-9\.]+)", line)
                if match:
                    step, ap = int(match.group(1)), float(match.group(2))
                    if ap > max_ap:
                        max_ap, max_step = ap, step
                        
    if max_ap != -1:
        print(f"Highest Segmentation AP: {max_ap} at Epoch: {max_step}")
    else:
        print("No segmentation AP information found in the log.")
    
    return max_step

if __name__ == "__main__":
    logfile = os.path.join(cfg.record_dir, 'recorder.log')
    best_model = find_highest_segm_ap(logfile)

    globals()["run_" + args.type]()
