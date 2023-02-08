import h5py
import os
import pdb
import numpy as np
import json
import sys
import requests

# FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]
FIELDNAMES = ['language', 'page_url', 'image_url', 'page_title','section_title','hierarchical_section_title',
'caption_reference_description','caption_attribution_description', 'caption_alt_text_description', 'mime_type', 
'original_height', 'original_width', 'is_main_image','attribution_passes_lang_id', 'page_changed_recently',
'context_page_description','context_section_description']
import csv
import base64
import pickle
import lmdb  # install lmdb by "pip install lmdb"
import click
import logging
import pandas as pd

@click.command()
@click.option("--input-dir", default=None, help="Project input directory.")
@click.option("--output-dir", default=None, help="Experiment output directory.")


def main(input_dir, output_dir):
    # Redirecting outputs to file
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, "output.log"), "a"))
    print = logger.info

    print("Starting conversion")

    csv.field_size_limit(sys.maxsize)

    count = 1
    en_item_count = 0
    infiles = []

    path = input_dir


    infiles.append(path + "/wit_v1.train.all-00000-of-00010.tsv")
    # infiles.append(path + "/wit_v1.train.all-00001-of-00010.tsv")

    save_path = os.path.join(
        output_dir, "wit.lmdb"
    )

    for infile in infiles:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(
                tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES
            )
            for item in reader:
                if item["language"] == "en":
                    print(item["page_url"])
                    r = requests.get(item["page_url"])
                    print(r.status_code)
                    print(r.encoding)
                    print(r.apparent_encoding)
                    en_item_count += 1
                    # print(item)

                if count % 10 == 0:
                    break
                    # print(count)
                count += 1
    print(en_item_count)
    print(count)
    # bs = [
    #     path2rest(path, split, annotations, label2ans) for path in tqdm(annot_paths)
    # ]

    # dataframe = pd.DataFrame(
    #     bs,
    #     columns=[
    #         "image",
    #         "questions",
    #         "answers",
    #         "answer_labels",
    #         "answer_scores",
    #         "image_id",
    #         "question_id",
    #         "split",
    #     ],
    # )

    # table = pa.Table.from_pandas(dataframe)

    # os.makedirs(output_dir, exist_ok=True)
    # with pa.OSFile(f"{output_dir}/wit.arrow", "wb") as sink:
    #     with pa.RecordBatchFileWriter(sink, table.schema) as writer:
    #         writer.write_table(table)



    # env = lmdb.open(save_path, map_size=1099511627776)

    # id_list = []
    # with env.begin(write=True) as txn:
    #     for infile in infiles:
    #         with open(infile) as tsv_in_file:
    #             reader = csv.DictReader(
    #                 tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES
    #             )
    #             for item in reader:
    #                 img_id = str(int(item["language"].split("_")[-1])).encode()
    #                 id_list.append(img_id)
    #                 # txn.put(img_id, pickle.dumps(item))
    #                 if count % 1000 == 0:
    #                     print(count)
    #                 count += 1
    #     txn.put("keys".encode(), pickle.dumps(id_list))

    # print(count)


if __name__ == "__main__":
    main()