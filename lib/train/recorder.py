import os
from collections import defaultdict, deque
import logging
import torch
from tensorboardX import SummaryWriter

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, cfg):
        log_dir = cfg.record_dir

        try:
            from tensorboardX import SummaryWriter
            if not cfg.resume:
                os.system("rm -rf {}".format(log_dir))
            self.writer = SummaryWriter(log_dir=log_dir)

            logging.basicConfig(filename='{}/recorder.log'.format(log_dir), level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

            self.use_tensorboard = False
        except ImportError:
            logging.basicConfig(filename='{}/recorder.log'.format(log_dir), level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
            
            logging.warning("tensorboardX not found, using logging instead.")
            self.use_tensorboard = False
        
        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        if "process_" + cfg.task in globals():
            self.processor = globals()["process_" + cfg.task]
        else:
            self.processor = None

    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        pattern = prefix + "/{}"
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats
        if self.use_tensorboard:
            for k, v in loss_stats.items():
                if isinstance(v, SmoothedValue):
                    self.writer.add_scalar(pattern.format(k), v.median, step)
                else:
                    self.writer.add_scalar(pattern.format(k), v, step)

            if self.processor is None:
                return
            image_stats = self.processor(image_stats) if image_stats else self.image_stats
            for k, v in image_stats.items():
                self.writer.add_image(pattern.format(k), v, step)
        else:
            # Logging instead of Tensorboard
            for k, v in loss_stats.items():
                logging.info(f'{pattern.format(k)} - Step: {step}, Value: {v.median if isinstance(v, SmoothedValue) else v}')        


    def state_dict(self):
        scalar_dict = {}
        scalar_dict["step"] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict["step"]

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append("{}: {:.4f}".format(k, v.avg))
        loss_state = "  ".join(loss_state)

        recording_state = "  ".join(
            ["epoch: {}", "step: {}", "{}", "data: {:.4f}", "batch: {:.4f}"]
        )
        return recording_state.format(
            self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg
        )


def make_recorder(cfg):
    return Recorder(cfg)
