import torch
import torch.nn as nn

from lib.utils import net_utils


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        # Detection stage
        self.ct_crit = net_utils.FocalLoss()
        self.radius_crit = net_utils.IndL1Loss1d("l1")
        self.reg_crit = net_utils.IndL1Loss1d("l1")

        # Segmentation stage
        self.py_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        output = self.net(batch["inp"], batch)

        scalar_stats = {}
        loss = 0

        ct_loss = self.ct_crit(output["ct_hm"], batch["ct_hm"])
        scalar_stats.update({"ct_loss": ct_loss})
        loss += ct_loss

        radius_loss = self.radius_crit(
            output["radius"], batch["radius"], batch["ct_ind"], batch["ct_01"]
        )
        scalar_stats.update({"radius_loss": radius_loss})
        loss += 0.1 * radius_loss

        reg_loss = self.reg_crit(output["reg"], batch["reg"], batch["ct_ind"], batch["ct_01"])
        scalar_stats.update({"reg_loss": reg_loss})
        loss += reg_loss

        py_loss = 0
        output["py_pred"] = [output["py_pred"][-1]]
        for i in range(len(output["py_pred"])):
            py_loss += self.py_crit(output["py_pred"][i], output["i_gt_py"]) / len(
                output["py_pred"]
            )
        scalar_stats.update({"py_loss": py_loss})
        loss += py_loss

        scalar_stats.update({"loss": loss})
        image_stats = {}

        # loss = [loss, py_loss]

        return output, loss, scalar_stats, image_stats
