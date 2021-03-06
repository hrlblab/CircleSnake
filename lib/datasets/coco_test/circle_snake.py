import math
import os

import cv2
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO

from lib.config import cfg
from lib.utils import data_utils
from lib.utils.snake import snake_coco_utils, snake_config, visualize_utils


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.anns = sorted(self.coco.getImgIds())
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

    def process_info(self, img_id):
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]["file_name"])
        return path, img_id

    def normalize_image(self, inp):
        inp = inp.astype(np.float32) / 255.0
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        ann = self.anns[index]

        path, img_id = self.process_info(ann)
        img = cv2.imread(path)

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])

        x = 32
        input_w = (int(width / 1.0) | (x - 1)) + 1
        input_h = (int(height / 1.0) | (x - 1)) + 1

        scale = np.array([input_w, input_h])

        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        # if cfg.debug_test:
        # cv2.imshow("Input", inp.astype(np.uint8))
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        if cfg.rotate == 90:
            inp = cv2.rotate(inp, cv2.ROTATE_90_CLOCKWISE)

        inp = self.normalize_image(inp)
        ret = {"inp": inp}
        meta = {"center": center, "scale": scale, "test": "", "img_id": img_id, "ann": ""}
        ret.update({"meta": meta})

        return ret

    def __len__(self):
        return len(self.anns)
