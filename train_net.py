import torch.multiprocessing

from lib.config import args, cfg
from lib.datasets import make_data_loader
from lib.evaluators import make_evaluator
from lib.networks import make_network
from lib.train import (make_lr_scheduler, make_optimizer, make_recorder,
                       make_trainer, set_lr_scheduler)
from lib.utils.net_utils import load_model, load_network, save_model


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    if cfg.segm_or_bbox == "both":
        cfg.segm_or_bbox = "segm"
        segm_evaluator = make_evaluator(cfg)
        cfg.segm_or_bbox = "bbox"
        det_evaluator = make_evaluator(cfg)
        cfg.segm_or_bbox = "both"
    else:
        evaluator = make_evaluator(cfg)

    begin_epoch = load_model(
        network,
        optimizer,
        scheduler,
        recorder,
        cfg.model_dir,
        resume=cfg.resume,
        pretrain=cfg.pretrain,
    )
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % cfg.eval_ep == 0:
            if cfg.segm_or_bbox == "both":
                trainer.val(epoch, val_loader, det_evaluator, recorder)
                trainer.val(epoch, val_loader, segm_evaluator, recorder)
            else:
                trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main():
    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    # Config is loaded when the method is called with all the right options
    main()
