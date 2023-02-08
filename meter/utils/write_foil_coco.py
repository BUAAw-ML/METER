import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(split, path, captions, info):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()

    return [binary, name, split, captions, info]


def make_arrow(root, image_root):

    annotations = {}
    images = {}

    with open(f"{root}/foilv1.0_train_2017.json", "r") as fp:
        annotations["train"] = json.load(fp)["annotations"]

    with open(f"{root}/foilv1.0_train_2017.json", "r") as fp:
        images["train"] = json.load(fp)["images"]
    
    with open(f"{root}/foilv1.0_test_2017.json", "r") as fp:
        annotations["val"] = json.load(fp)["annotations"]

    with open(f"{root}/foilv1.0_test_2017.json", "r") as fp:
        images["val"] = json.load(fp)["images"]


    for split in [
        "train",
        "val"
    ]:

        split_name = {
            "train": "train2014",
            "val": "val2014"
        }[split]

        paths = list(glob(f"{image_root}/{split_name}/*.jpg"))
        print(f'The number of images in image_root: {len(paths)}')

        id2file = defaultdict(dict)
        for img in tqdm(images[split]):
            id2file[img["id"]] = img["file_name"]

        # bs = [
        #     path2rest(item, split, f"{image_root}/{split_name}/{id2file[item['image_id']]}")
        #     for item in tqdm(annotations[split])
        #     if f"{image_root}/{split_name}/{id2file[item['image_id']]}" in paths
        # ]

        bs = []
        cur_image_id = -1
        for item in tqdm(annotations[split]):
            if f"{image_root}/{split_name}/{id2file[item['image_id']]}" not in paths:
                continue
            if item['image_id'] != cur_image_id:
                if cur_image_id != -1:
                    bs.append(path2rest(split, f"{image_root}/{split_name}/{id2file[item['image_id']]}", captions, info))

                cur_image_id = item['image_id']
                captions = []
                info = []
            
            captions.append(item["caption"])
            info.append({'target_word': item["target_word"], 'foil_word': item["foil_word"], 'foil': item["foil"]})


        
        print(f'The number of samples: {len(bs)}')

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "name",
                "split",
                "caption",
                "info",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(root, exist_ok=True)
        with pa.OSFile(f"{root}/foilcoco_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

make_arrow(root='/data/qbwang/METER/data/pretrain_data/foil_coco', image_root='/data/datasets/coco_2014')