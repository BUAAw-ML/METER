import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os
from tqdm import tqdm
from glob import glob

def path2rest(path, iid2captions, split):
    iid = path.split("/")[-1].split(".")[0]
    
    with open(path, "rb") as fp:
        binary = fp.read()

    captions = iid2captions[iid]
    
    return [
        binary,
        captions,
        iid,
        split,
    ]

def make_arrow(root, targetroot):
    
    with open(f"{root}/txt_mapper.json", "r") as fp:
        txt_mapper = json.load(fp)
        
    for split in ["train"]:#,val
        with open(f"{root}/{split}_id.json", "r") as fp:
            ids = json.load(fp)
            
        iid2captions = dict()
        for iid in tqdm(ids):
            if iid in txt_mapper:
                iid2captions[iid] = [txt_mapper[iid]]
            
        paths = list(glob(f"{root}/gcc3m_img_val/*.jpg"))#
        random.shuffle(paths)

        # for path in paths:
        #     if path.split("/")[-1].split(".")[0] in iid2captions:
        #         print(path.split("/")[-1].split(".")[0])
        
        # exit()
        
        caption_paths = [path for path in paths if path.split("/")[-1].split(".")[0] in iid2captions]
        print(
            len(paths), len(caption_paths), len(iid2captions),
        )

        sub_len = int(len(caption_paths) // 100000)
        subs = list(range(sub_len + 1))
        for sub in subs:
            sub_paths = caption_paths[sub * 100000 : (sub + 1) * 100000]
            bs = [path2rest(path, iid2captions, split) for path in tqdm(sub_paths)]
            dataframe = pd.DataFrame(
                bs, columns=["image", "caption", "image_id", "split"],
            )
            table = pa.Table.from_pandas(dataframe)

            with pa.OSFile(
                f"{targetroot}/conceptual_caption_{split}_{sub}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del bs
            gc.collect()
