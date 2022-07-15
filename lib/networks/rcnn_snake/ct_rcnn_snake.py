import os

import torch
import torch.nn as nn

from lib.config import cfg
from lib.utils import data_utils, net_utils
from lib.utils.snake import snake_decode

from .cp_head import ComponentDetection
from .dla import DLASeg
from .evolve import Evolution


class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=""):
        super(Network, self).__init__()

        # Backbone network
        self.dla = DLASeg(
            "dla{}".format(num_layers),
            heads,
            pretrained=True,
            down_ratio=down_ratio,
            final_kernel=1,
            last_level=5,
            head_conv=head_conv,
        )
        # Center, height, width detection
        self.cp = ComponentDetection()
        # Evolve the initial contour
        self.gcn = Evolution()

        det_dir = os.path.join(os.path.dirname(cfg.model_dir), cfg.det_model)
        net_utils.load_network(self, det_dir, strict=False)

    def decode_detection(self, output, h, w):
        ct_hm = output["act_hm"]
        wh = output["awh"]
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({"ct": ct, "detection": detection})
        return ct, detection

    def forward(self, x, batch=None):
        output, cnn_feature = self.dla(x)
        with torch.no_grad():
            self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
        output = self.cp(output, cnn_feature, batch)
        output = self.gcn(output, cnn_feature, batch)
        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=""):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
