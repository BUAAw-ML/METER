import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word


def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


def path2rest(path, split, annotations, label2ans):
    iid = int(path.split("/")[-1][:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas]
    answer_labels = (
        [a["labels"] for a in answers]
    )
    answer_scores = (
        [a["scores"] for a in answers]
    )
    answers = (
        [[label2ans[l] for l in al] for al in answer_labels]
    )

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, split]


def make_arrow(root, dataset_root, image_root):
    with open(f"{root}/aokvqa_v1p0_train.json", "r") as fp:
        questions_train = json.load(fp)
    with open(f"{root}/aokvqa_v1p0_val.json", "r") as fp:
        questions_val = json.load(fp)
    # with open(f"{root}/aokvqa_v1p0_test.json", "r") as fp:
    #     questions_test = json.load(fp)


    all_major_answers = list()
    annotations = dict()
    for split, item in zip(
        ["train", "val"],
        [
            questions_train,
            questions_val
            # questions_test
        ],
    ):
        _annot = defaultdict(dict)
        for q in tqdm(item):

            _annot[q["image_id"]][q["question_id"]] = [q["question"]]
            annotations[split] = _annot
            answers = q["direct_answers"]
            count_answers = {}
            
            for answer in answers:
                
                if answer in count_answers.keys():
                    count_answers[answer] += 1
                else:
                    count_answers[answer] = 1

            gtruth = max(count_answers.keys(), key=(lambda key: count_answers[key]))

            all_major_answers.append(gtruth)
        
    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}#9}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())
    


    for split, item in zip(
        ["train", "val"],
        [
            questions_train,
            questions_val
        ],
    ):
        _annot = annotations[split]
        for q in tqdm(item):
            answers = q["direct_answers"]
            answer_count = {}
            for answer_ in answers:
                answer_count[answer_] = answer_count.get(answer_, 0) + 1

            labels = []
            scores = []
            for answer in answer_count:
                if answer not in ans2label:
                    continue
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)

            _annot[q["image_id"]][q["question_id"]].append(
                {"labels": labels, "scores": scores,}
            )

    for split in ["train", "val"]:
        filtered_annot = dict()
        for ik, iv in annotations[split].items():
            new_q = dict()
            for qk, qv in iv.items():
                if len(qv[1]["labels"]) != 0:
                    new_q[qk] = qv
            if len(new_q) != 0:
                filtered_annot[ik] = new_q
        annotations[split] = filtered_annot

        # question_num  =  0
        # for _, iv in filtered_annot.items():
        #     question_num += len(iv)
        # print(question_num)

    for split in [
        "train",
        "val"
    ]:
        annot = annotations[split]
        split_name = {
            "train": "train2017",
            "val": "val2017"
        }[split]

        paths = list(glob(f"{image_root}/{split_name}/*.jpg"))
        random.shuffle(paths)

        annot_paths = [
            path
            for path in paths
            if int(path.split("/")[-1][:-4]) in annot
        ]

        if len(paths) == len(annot_paths):
            print("use all images")
        else:
            print("not use all images")
        print(
            len(paths), len(annot_paths), len(annot),
        )

        # print(label2ans)
        # print(len(label2ans))
        # exit()

        bs = [
            path2rest(path, split, annotations, label2ans) for path in tqdm(annot_paths)
        ]

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/aokvqa_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

    # table = pa.ipc.RecordBatchFileReader(
    #     pa.memory_map(f"{dataset_root}/okvqa_val.arrow", "r")
    # ).read_all()

    # pdtable = table.to_pandas()

    # df1 = pdtable[:-1000]
    # df2 = pdtable[-1000:]

    # df1 = pa.Table.from_pandas(df1)
    # df2 = pa.Table.from_pandas(df2)

    # with pa.OSFile(f"{dataset_root}/okvqa_trainable_val.arrow", "wb") as sink:
    #     with pa.RecordBatchFileWriter(sink, df1.schema) as writer:
    #         writer.write_table(df1)

    # with pa.OSFile(f"{dataset_root}/okvqa_rest_val.arrow", "wb") as sink:
    #     with pa.RecordBatchFileWriter(sink, df2.schema) as writer:
    #         writer.write_table(df2)
