import json
import math
import os

import cv2
import numpy as np
import pycocotools.coco as coco
import pycocotools.mask as mask_util
from PIL import Image
from pycocotools.cocoeval import COCOeval

import lib.utils.circle.kidpath_circle as kidpath_circle
from external.cityscapesscripts.evaluation import \
    evalInstanceLevelSemanticLabeling
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils
from lib.utils.circle.circle_eval import CIRCLEeval
from lib.utils.snake import (snake_cityscapes_utils, snake_config,
                             snake_eval_utils, snake_poly_utils,
                             visualize_utils)


class Evaluator:
    def __init__(self, result_dir):
        self.results = []
        self.rotate_results = []
        self.img_ids = []
        self.aps = []
        self.dices = []
        self.rotate_dices = []

        self.result_dir = result_dir
        os.system("mkdir -p {}".format(self.result_dir))
        if cfg.use_val:
            args = DatasetCatalog.get(cfg.val.dataset)
        else:
            args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args["ann_file"]
        self.data_root = args["data_root"]
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
            }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.iter_num = 0
        self.dice = 0
        self.num_images = 0

        self.mask = []
        self.rotate_mask = []

        self.color_map = {
            0: (0, 255, 0),     
            1: (0, 255, 255),   
            2: (255, 0, 255),   
            3: (255, 255, 0),
            4: (240,248,255),
            5: (152,245,255),
            6: (255,97,3), 
            7: (205,51,51), 
            8: (139,35,35), 
            9: (222,184,135), 
            10: (127,255,212), 
            11: (224,238,238), 
            12: (205,155,29),     
            # You can keep adding more category-color pairs here as needed
        }

    def evaluate(self, output, batch):
        detection = output["detection"]
        score = detection[:, 3].detach().cpu().numpy()
        label = detection[:, 4].detach().cpu().numpy().astype(int)
        py = output["py"][-1].detach().cpu().numpy() * snake_config.down_ratio

        img_id = int(batch["meta"]["img_id"][0])
        center = batch["meta"]["center"][0].detach().cpu().numpy()
        scale = batch["meta"]["scale"][0].detach().cpu().numpy()

        h, w = batch["inp"].size(2), batch["inp"].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img["height"], img["width"]
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)

        path = os.path.join(self.data_root, img["file_name"])
        orig_img = cv2.imread(path)
        cnt = 0


        if cfg.debug_test:
            # Prediction
            pred_img = orig_img.copy()

            for poly_idx, polys in enumerate(py):
                category = label[cnt]
                color = self.color_map.get(category, (0, 0, 0))  # default color is black if category is not in the map
                poly_corrected = np.zeros(shape=(128, 2), dtype=np.int32)

                # Limit to border
                for i, (poly_x, poly_y) in enumerate(polys):
                    if poly_x < 0:
                        poly_x = 0
                    elif poly_x > ori_w:
                        poly_x = ori_w
                    if poly_y < 0:
                        poly_y = 0
                    elif poly_y > ori_h:
                        poly_y = ori_h
                    poly_corrected[i] = int(round(poly_x)), int(round(poly_y))
                cv2.polylines(pred_img, [poly_corrected], True, color, 2)
                text_pt_x = min(np.array(poly_corrected)[:, 0])
                text_pt_y = min(np.array(poly_corrected)[:, 1])
                # cv2.putText(pred_img, "%.2f" % score[poly_idx], (text_pt_x, text_pt_y), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (0, 0, 0), 1,)
                cnt += 1

            # Ground truth
            gt_img = orig_img.copy()
            ann_ids = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                # print(ann["segmentation"])
                instance_poly = [
                    np.array(poly, dtype=int).reshape(-1, 2) for poly in ann["segmentation"]
                ]

                if len(instance_poly) > 1:
                    instance_poly = instance_poly[:1]

                category = ann["category_id"]
                color = self.color_map.get(category, (0, 0, 0))
                cv2.polylines(gt_img, instance_poly, True, color, 2)

                text_pt_x = min(np.array(instance_poly)[0, :, 0])
                text_pt_y = min(np.array(instance_poly)[0, :, 1])
                # cv2.putText(gt_img, "%.2f" % 1, (text_pt_x, text_pt_y), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (0, 0, 0), 1)
                

            if cfg.save_images:
                path = os.path.join("/home/cnet/Journal_CodeWithData/CircleSnake/output_img/{}"
                                    .format(str(self.data_root).split('/')[-2]), img["file_name"])
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path, "circlesnake_pred_segm.png"), pred_img)
                cv2.imwrite(os.path.join(path, "circlesnake_truth_segm.png"), gt_img)

            if cfg.show_images:
                cv2.imshow("Prediction", pred_img)
                cv2.imshow("GT", gt_img)
                cv2.waitKey(0)

        if cfg.dice:
            # Prediction mask
            shape = (orig_img.shape[0],orig_img.shape[1])

            pred_mask = np.zeros(shape, dtype=np.uint8)
            for polys in py:
                cv2.drawContours(pred_mask, [polys.astype(int)], -1, 1, -1)

            if cfg.debug_test and cfg.show_images:
                cv2.imshow("Pred Mask", pred_mask)

            # GT Mask
            gt_mask = np.zeros(shape, dtype=np.uint8)

            ann_ids = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                instance_poly = [
                    np.array(poly, dtype=int).reshape(-1, 2) for poly in ann["segmentation"]
                ]
                cv2.drawContours(gt_mask, instance_poly, -1, 1, -1)

            if cfg.debug_test and cfg.show_images:
                cv2.imshow("Truth Mask", gt_mask)


            intersection = np.sum(np.logical_and(gt_mask, pred_mask))
            denominator = np.sum(gt_mask) + np.sum(pred_mask)

            if denominator == 0:
                if np.sum(gt_mask) == np.sum(pred_mask):
                    dice_score = 1
                else:
                    dice_score = 0
            else:
                dice_score = 2 * intersection / denominator

            self.dices.append(dice_score)

            self.dice += dice_score
            self.num_images += 1

            if cfg.debug_test and cfg.show_images:
                cv2.waitKey(0)


        self.iter_num += 1

        coco_dets = []
        for i in range(len(rles)):
            detection = {
                "image_id": img_id,
                "category_id": self.contiguous_category_id_to_json_id[label[i]],
                "segmentation": rles[i],
                "score": float("{:.2f}".format(score[i])),
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def evaluate_rotate(self, output, batch, rotate=False):
        detection = output["detection"]
        score = detection[:, 3].detach().cpu().numpy()
        label = detection[:, 4].detach().cpu().numpy().astype(int)
        py = output["py"][-1].detach().cpu().numpy() * snake_config.down_ratio

        img_id = int(batch["meta"]["img_id"][0])
        center = batch["meta"]["center"][0].detach().cpu().numpy()
        scale = batch["meta"]["scale"][0].detach().cpu().numpy()

        h, w = batch["inp"].size(2), batch["inp"].size(3)

        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img["height"], img["width"]
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)

        path = os.path.join(self.data_root, img["file_name"])
        orig_img = cv2.imread(path)
        cnt = 0

        if cfg.debug_test:
            # Prediction
            pred_img = orig_img.copy()
            pred_img = cv2.rotate(pred_img, cv2.ROTATE_90_CLOCKWISE)

            for poly_idx, polys in enumerate(py):
                category = label[cnt]
                color = self.color_map.get(category, (0, 0, 0))  # default color is black if category is not in the map
                poly_corrected = np.zeros(shape=(128, 2), dtype=np.int32)

                # Limit to border
                for i, (poly_x, poly_y) in enumerate(polys):
                    if poly_x < 0:
                        poly_x = 0
                    elif poly_x > ori_w:
                        poly_x = ori_w
                    if poly_y < 0:
                        poly_y = 0
                    elif poly_y > ori_h:
                        poly_y = ori_h
                    poly_corrected[i] = int(round(poly_x)), int(round(poly_y))
                cv2.polylines(pred_img, [poly_corrected], True, color, 2)
                text_pt_x = min(np.array(poly_corrected)[:, 0])
                text_pt_y = min(np.array(poly_corrected)[:, 1])
                # cv2.putText(pred_img, "%.2f" % score[poly_idx], (text_pt_x, text_pt_y), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (0, 0, 0), 1,)
                cnt += 1

            if cfg.save_images:
                path = os.path.join("/home/cnet/Journal_CodeWithData/CircleSnake/output_img/{}"
                                    .format(str(self.data_root).split('/')[-2]), img["file_name"])
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path, "circlesnake_pred_rotate_segm.png"), pred_img)

        # Prediction mask
        if cfg.dice:
            shape = (orig_img.shape[0],orig_img.shape[1])
            pred_mask = np.zeros(shape, dtype=np.uint8)
            for polys in py:
                cv2.drawContours(pred_mask, [polys.astype(int)], -1, 1, -1)

            pred_mask = np.rot90(pred_mask)

        # GT Mask
            gt_mask = np.zeros(shape, dtype=np.uint8)
            ann_ids = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                instance_poly = [
                    np.array(poly, dtype=int).reshape(-1, 2) for poly in ann["segmentation"]
                ]
                cv2.drawContours(gt_mask, instance_poly, -1, 1, -1)

            intersection = np.sum(np.logical_and(gt_mask, pred_mask))
            denominator = np.sum(gt_mask) + np.sum(pred_mask)

            if denominator == 0:
                if np.sum(gt_mask) == np.sum(pred_mask):
                    dice_score = 1
                else:
                    dice_score = 0
            else:
                dice_score = 2 * intersection / denominator

            self.rotate_dices.append(dice_score)

            self.dice += dice_score
            self.num_images += 1

            if  cfg.save_images is True:
                path = os.path.join("/home/cnet/Journal_CodeWithData/CircleSnake/output_img/{}"
                                    .format(str(self.data_root).split('/')[-2]), img["file_name"])
                pred_mask = cv2.rotate(pred_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(os.path.join(path, 'rotate_pred_mask.png'), pred_mask)
    
        self.iter_num += 1


        rotate_pred = []
        for i in range(len(rles)):
            detection = {
                "image_id": img_id,
                "category_id": self.contiguous_category_id_to_json_id[label[i]],
                "segmentation": rles[i],
                "score": float("{:.2f}".format(score[i])),
            }
            rotate_pred.append(detection)

        self.rotate_results.extend(rotate_pred)
        self.img_ids.append(img_id)

    def summarize_rotate(self):
        json.dump(self.rotate_results, open(os.path.join(self.result_dir, "rotate_results.json"), "w"))
        json.dump(self.rotate_dices, open(os.path.join(self.result_dir, "rotate_dice.json"), "w"))
        self.dice /= self.num_images
        print("Rotate Dice", self.dice)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, "results.json"), "w"))
        json.dump(self.dices, open(os.path.join(self.result_dir, "dice.json"), "w"))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, "results.json"))
        coco_eval = COCOeval(self.coco, coco_dets, "segm")
        # coco_eval.params.maxDets = [1000, 1000, 1000]
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        if cfg.dice:
            self.dice /= self.num_images
            print("Dice Score:", self.dice)
        return {"segm_ap": coco_eval.stats[0]}


class DetectionEvaluator:
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.aps = []
        self._valid_ids = [1]

        self.result_dir = result_dir
        os.system("mkdir -p {}".format(self.result_dir))

        if cfg.use_val:
            args = DatasetCatalog.get(cfg.val.dataset)
        else:
            args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args["ann_file"]
        self.data_root = args["data_root"]
        self.coco = coco.COCO(self.ann_file)
        self.circle = kidpath_circle.CIRCLE(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.circle.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.iter_num = 0

        self.color_map = {
            0: (0, 255, 0),     
            1: (0, 255, 255),   
            2: (255, 0, 255),   
            3: (255, 255, 0),
            4: (240,248,255),
            5: (152,245,255),
            6: (255,97,3), 
            7: (205,51,51), 
            8: (139,35,35), 
            9: (222,184,135), 
            10: (127,255,212), 
            11: (224,238,238), 
            12: (205,155,29),     
            # You can keep adding more category-color pairs here as needed
        }

    def evaluate(self, output, batch):
        detection = output["detection"]
        detection = detection[0] if detection.dim() == 3 else detection
        circle = detection[:, :3].detach().cpu().numpy() * snake_config.down_ratio
        score = detection[:, 3].detach().cpu().numpy()
        label = detection[:, 4].detach().cpu().numpy().astype(int)

        img_id = int(batch["meta"]["img_id"][0])
        center = batch["meta"]["center"][0].detach().cpu().numpy()
        scale = batch["meta"]["scale"][0].detach().cpu().numpy()

        if len(circle) == 0:
            return

        h, w = batch["inp"].size(2), batch["inp"].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.circle.loadImgs(img_id)[0]
        ori_h, ori_w = img["height"], img["width"]

        if cfg.debug_test:
            # Read the image
            path = os.path.join(self.data_root, img["file_name"])
            pred_img = cv2.imread(path)
            gt_img = pred_img.copy()

            # Overlay the prediction in green
            for i in range(len(label)):
                circle_ = data_utils.affine_transform(circle[i][:2].reshape(-1, 2), trans_output_inv).ravel()
                circle_[0] = np.clip(circle_[0], 0, 512 - 1)
                circle_[1] = np.clip(circle_[1], 0, 512 - 1)
                radius = max(0, int(circle[i][2] * trans_output_inv[0][0]))
                color = self.color_map.get(label[i], (0, 0, 0))
                cv2.circle(pred_img, (int(circle_[0]), int(circle_[1])), radius, color, 2)
                # cv2.putText(pred_img, str(label[i]), (int(circle_[0]), int(circle_[1]-radius)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (0, 0, 0), 1,)

            # cv2.circle(pred_img, (100, 0), 20,
            #            (0, 255, 0), 2)

            # Show the image
            # cv2.imshow("Prediction", pred_img)

            # Ground truth
            ann_ids = self.circle.getAnnIds(img_id)
            anns = self.circle.loadAnns(ann_ids)
            for ann in anns:
                radius = ann["circle_radius"]
                x, y = ann["circle_center"][0], ann["circle_center"][1]
                category = ann["category_id"]
                color = self.color_map.get(category, (0, 0, 0))
                cv2.circle(gt_img, (int(x),int(y)), int(radius), color, 2)
                # cv2.putText(gt_img, str(label[i]), (int(x), int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (0, 0, 0), 1,)
        
            # cv2.imshow("GT", gt_img)
            if cfg.save_images:
                path = os.path.join("/home/cnet/Journal_CodeWithData/CircleSnake/output_img/{}"
                                    .format(str(self.data_root).split('/')[-2]), img["file_name"])
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path, "circlesnake_pred_det.png"), pred_img)
                cv2.imwrite(os.path.join(path, "circlesnake_truth_det.png"), gt_img)

            self.iter_num += 1
            # cv2.waitKey(0)

        circle_dets = []
        for i in range(len(label)):
            circle_ = data_utils.affine_transform(
                circle[i][:2].reshape(-1, 2), trans_output_inv
            ).ravel()
            circle_ = list(map(lambda x: float("{:.2f}".format(x)), circle_))
            detection = {
                "image_id": img_id,
                "category_id": self.contiguous_category_id_to_json_id[label[i]],
                "score": float("{:.2f}".format(score[i])),
                "circle_center": [circle_[0], circle_[1]],
                "circle_radius": self._to_float(circle[i][2]),
            }
            circle_dets.append(detection)

        self.results.extend(circle_dets)
        self.img_ids.append(img_id)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_circle_format(self, all_circles):
        detections = []
        for image_id in all_circles:
            for cls_ind in all_circles[image_id]:
                try:
                    category_id = self._valid_ids[cls_ind - 1]
                except:
                    aaa = 1
                for circle in all_circles[image_id][cls_ind]:
                    score = circle[3]
                    circle_out = list(map(self._to_float, circle[0:3]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "score": float("{:.2f}".format(score)),
                        "circle_center": [circle_out[0], circle_out[1]],
                        "circle_radius": circle_out[2],
                    }
                    if len(circle) > 5:
                        extreme_points = list(map(self._to_float, circle[5:13]))
                        detection["extreme_points"] = extreme_points

                    # output_h = 512  # hard coded
                    # output_w = 512  # hard coded
                    # cp = [0, 0]
                    # cp[0] = circle_out[0]
                    # cp[1] = circle_out[1]
                    # cr = circle_out[2]
                    # if cp[0] - cr < 0 or cp[0] + cr > output_w:
                    #     continue
                    # if cp[1] - cr < 0 or cp[1] + cr > output_h:
                    #     continue

                    detections.append(detection)
        return detections

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, "results.json"), "w"))
        circle_dets = self.circle.loadRes(os.path.join(self.result_dir, "results.json"))
        circle_eval = CIRCLEeval(self.circle, circle_dets, "circle")
        circle_eval.params.imgIds = self.img_ids
        # circle_eval.params.maxDets = [1000, 1000, 1000]
        circle_eval.evaluate()
        circle_eval.accumulate()
        circle_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(circle_eval.stats[0])
        return {"det_ap": circle_eval.stats[0]}

Evaluator = Evaluator if cfg.segm_or_bbox == "segm" else DetectionEvaluator
