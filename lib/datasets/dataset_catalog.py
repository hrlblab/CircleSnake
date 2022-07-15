from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        "monusegTrain": {
            "id": "coco",
            "data_root": "data/monuseg/train",
            "ann_file": "data/monuseg/MoNuSeg_train2021.json",
            "split": "train",
        },
        "monusegVal": {
            "id": "coco",
            "data_root": "data/monuseg/val",
            "ann_file": "data/monuseg/MoNuSeg_val2021.json",
            "split": "test",
        },
        "monusegTest": {
            "id": "coco_test",
            "data_root": "data/monuseg/test",
            "ann_file": "data/monuseg/MoNuSeg_test2021.json",
            "split": "test",
        },
        "CocoTrain": {
            "id": "coco",
            "data_root": "data/kidpath_coco/train",
            "ann_file": "data/kidpath_coco/train_circle.json",
            "split": "train",
        },
        "CocoVal": {
            "id": "coco",
            "data_root": "data/kidpath_coco/validate",
            "ann_file": "data/kidpath_coco/validate_circle.json",
            "split": "test",
        },
        "CocoMini": {
            "id": "coco",
            "data_root": "data/coco/val2017",
            "ann_file": "data/coco/annotations/instances_val2017.json",
            "split": "mini",
        },
        "CocoTest": {
            "id": "coco_test",
            "data_root": "data/kidpath_coco/test",
            "ann_file": "data/kidpath_coco/test_circle.json",
            "split": "test",
        },
        "CityscapesTrain": {
            "id": "cityscapes",
            "data_root": "data/cityscapes/leftImg8bit",
            "ann_file": (
                "data/cityscapes/annotations/train",
                "data/cityscapes/annotations/train_val",
            ),
            "split": "train",
        },
        "CityscapesVal": {
            "id": "cityscapes",
            "data_root": "data/cityscapes/leftImg8bit",
            "ann_file": "data/cityscapes/annotations/val",
            "split": "val",
        },
        "CityscapesCocoVal": {
            "id": "cityscapes_coco",
            "data_root": "data/cityscapes/leftImg8bit/val",
            "ann_file": "data/cityscapes/coco_ann/instance_val.json",
            "split": "val",
        },
        "CityCocoBox": {
            "id": "cityscapes_coco",
            "data_root": "data/cityscapes/leftImg8bit/val",
            "ann_file": "data/cityscapes/coco_ann/instance_box_val.json",
            "split": "val",
        },
        "CityscapesMini": {
            "id": "cityscapes",
            "data_root": "data/cityscapes/leftImg8bit",
            "ann_file": "data/cityscapes/annotations/val",
            "split": "mini",
        },
        "CityscapesTest": {
            "id": "cityscapes_test",
            "data_root": "data/cityscapes/leftImg8bit/test",
        },
        "SbdTrain": {
            "id": "sbd",
            "data_root": "data/sbd/img",
            "ann_file": "data/sbd/annotations/sbd_train_instance.json",
            "split": "train",
        },
        "SbdVal": {
            "id": "sbd",
            "data_root": "data/sbd/img",
            "ann_file": "data/sbd/annotations/sbd_trainval_instance.json",
            "split": "val",
        },
        "SbdMini": {
            "id": "sbd",
            "data_root": "data/sbd/img",
            "ann_file": "data/sbd/annotations/sbd_trainval_instance.json",
            "split": "mini",
        },
        "VocVal": {
            "id": "voc",
            "data_root": "data/voc/JPEGImages",
            "ann_file": "data/voc/annotations/voc_val_instance.json",
            "split": "val",
        },
        "KinsTrain": {
            "id": "kins",
            "data_root": "data/kitti/training/image_2",
            "ann_file": "data/kitti/training/instances_train.json",
            "split": "train",
        },
        "KinsVal": {
            "id": "kins",
            "data_root": "data/kitti/testing/image_2",
            "ann_file": "data/kitti/testing/instances_val.json",
            "split": "val",
        },
        "KinsMini": {
            "id": "kins",
            "data_root": "data/kitti/testing/image_2",
            "ann_file": "data/kitti/testing/instances_val.json",
            "split": "mini",
        },
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
