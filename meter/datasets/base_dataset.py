from genericpath import exists
import random
import torch
import io
import pyarrow as pa
import os
import numpy as np
from PIL import Image
from ..transforms import keys_to_transforms
import re
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizer
import tqdm
import pickle
import pandas as pd

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
    RobertaTokenizer,
    AutoTokenizer,
    T5Tokenizer,
)


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        tokenizer=None,
        masking_strategy='token_masking'
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.clip_transform = False
        for transform_key in transform_keys:
            if 'clip' in transform_key:
                self.clip_transform = True
                break
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir

        self.masking_strategy = masking_strategy

        self.entity_mlm_probability = 0.0
        print(f'entity_mlm_probability: {self.entity_mlm_probability}!')

        print(f'load datasets: {names}!')
        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]
            
            # for i, name in enumerate(names):
            #     if "wit" in name:
            #         print(f"Process {name} caption!")
            #         pandas_table = tables[i].to_pandas()

            #         for index,_ in pandas_table.iterrows():
            #             pandas_table.at[index,'caption'] = [pandas_table.at[index,'caption'][0].split('#')[0]]

            #         tables[i] = pa.Table.from_pandas(pandas_table)


            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])
            self.table = pa.concat_tables(tables, promote=True)

            print(f'Columns of data table!:{self.table.to_pandas().columns.values}')

            # pd.set_option('max_colwidth',1000)
            # print(self.table.to_pandas()[:20][['caption','caption_entities']])
            # exit()

            print(f'Data table size: {len(self.table)}!')
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                if self.masking_strategy == 'entity_masking':
                    self.text_entities = self.table['caption_entities'].to_pandas().tolist()

                if type(self.all_texts[0][0]) == str:
                    self.all_texts = (
                        [list(set(texts)) for texts in self.all_texts]
                        if remove_duplicate
                        else self.all_texts
                    )
                else: #snli
                    self.all_texts = (
                        [[t[1].strip() for t in texts] for texts in self.all_texts]
                    )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        if text_column_name != "" and not self.image_only:
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)): 
                self.index_mapper[i] = (i, None)
        


        if self.masking_strategy == 'entity_masking':
            self.tokenizer = tokenizer#BertTokenizer.from_pretrained("bert-base-uncased")#tokenizer

            self.texts_entity_mask_ratio = 0
            self.texts_num = 0

            if 'roberta' in tokenizer.name_or_path:
                self.text_entities_mask = self.get_entities_roberta()

            elif 'bert' in tokenizer.name_or_path:
                self.text_entities_mask = self.get_entities_bert()
            else:
                raise NotImplementedError()
            # data_entities_file = f'{data_dir}/{len(self.table)}.pkl'
            # if exists(data_entities_file):
            #     self.text_entities = pickle.load(open(data_entities_file, 'r'))
            # else:
            #     self.text_entities = self.get_entities()
            #     pickle.dump(self.text_entities, open(data_entities_file, 'w'))
            print(f'Average masking ratio: { self.texts_entity_mask_ratio / self.texts_num}!')
            # exit()


    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())

        image_bytes.seek(0)
        if self.clip_transform:
            return Image.open(image_bytes).convert("RGBA")
        else:
            return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )

        # encoding["raw_index"] = raw_index
        # print(self.split)
        # print(self.masking_strategy)
        if self.masking_strategy == 'entity_masking':
            return {
                "text": (text, encoding, self.text_entities_mask[index][caption_index]),
                "img_index": index,
                "cap_index": caption_index,
                "raw_index": raw_index
            }
        else:

            return {
                "text": (text, encoding),
                "img_index": index,
                "cap_index": caption_index,
                "raw_index": raw_index
            }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index, caption_index = self.index_mapper[random_index]
        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        if self.masking_strategy == 'entity_masking':
            return {f"false_text_{rep}": (text, encoding, self.text_entities_mask[index][caption_index])}
        else:
            return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)

                # index, caption_index = self.index_mapper[index]
                # text = self.all_texts[index][caption_index]
                # print(text)
        return ret

    def collate(self, batch, mlm_collator):

        # print(isinstance(mlm_collator, DataCollatorForWholeWordMask))

        batch_size = len(batch)

        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]

            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]

            
            flatten_mlms = mlm_collator(flatten_encodings)
            # print(mlm_collator.mlm)
            if len(dict_batch["text"][0])==3: #self.masking_strategy == 'entity_masking':# and self.split == 'train':
                
                # print(self.tokenizer.tokenize(dict_batch["text"][0][0]))
                text_entities_mask = [[d[2] for d in dict_batch[txt_key]] for txt_key in txt_keys]
                # print(text_entities_mask)
                flatten_text_entities_mask  = np.array([e for text_entitie_mask  in text_entities_mask  for e in text_entitie_mask])
                # print(flatten_mlms["labels"].shape)
                # print(torch.tensor(flatten_text_entities_mask).shape)
                flatten_mlms["labels"].masked_fill_(torch.tensor(flatten_text_entities_mask), value=-100)
                flatten_mlms["input_ids"].masked_fill_(torch.tensor(1- flatten_text_entities_mask), value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token))
                # print(flatten_mlms["input_ids"])
                # print(flatten_mlms["labels"])
                # exit()

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch


    def get_entities_roberta(self):
        
        text_entities_mask = []
        for index, item in enumerate(tqdm.tqdm(self.all_texts, desc = "Extract entities!")):
            item_entities_mask = []

            # for _, entity in enumerate(self.text_entities[index]):
            #     encoding_entities_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity)))

            for caption_index, _ in enumerate(item):
                
                text = self.all_texts[index][caption_index]
                # print(text)
                # print(self.tokenizer.tokenize(text))
                encoding = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_len,
                    return_special_tokens_mask=True,
                )

                item_entities_mask.append(self.exclude_non_entities_roberta(encoding["input_ids"], self.text_entities[index], encoding["special_tokens_mask"]))
            text_entities_mask.append(item_entities_mask)
            # print(item_entities_mask)
            # exit()

        return text_entities_mask 


    def exclude_non_entities_roberta(self, inputs, entities, special_tokens_mask):
        """Exclude non-entity tokens from token masking."""
        # print(entities)
        # print(len(entities))
        sentence_entities_positions = []
        # sign =False
        # count = 0
        for entity_text in entities:
            # entity_text= entity_text.replace(' -','-')
            # entity_text= entity_text.replace('- ','-')
            for i in range(len(inputs)):
                if inputs[i] == 2:
                    break
                for token_num in range(1, 4):
                    cur_tokens_text = self.tokenizer.decode(inputs[i : i + token_num])
                    if cur_tokens_text.lower().strip() == entity_text.lower().strip():
                        #   or \
                        # cur_tokens_text.lower().strip() == entity_text.lower().strip()+'s' or \
                        # cur_tokens_text.lower().strip() == entity_text.lower().strip()+'es' or \
                        # cur_tokens_text.lower().strip() == entity_text.lower().strip()[:-1]+'es'
                        # print(self.tokenizer.decode(entity_id))
                        # print(cur_tokens_text)
                        sequence_ids = list(range(i, i + token_num))

                        if random.random() >= self.entity_mlm_probability:
                            sentence_entities_positions.extend(sequence_ids)
                        # sign = True
                        break
                     
                    elif cur_tokens_text.lower().strip() not in entity_text.lower().strip():
                        break
        self.texts_entity_mask_ratio += len(sentence_entities_positions)/len(special_tokens_mask)
        self.texts_num += 1
        #     if sign == True:
        #         count += 1
        # print(count / len(entities))

        # print(len(sentence_entities_positions))
        new_stm = []
        for idx, _ in enumerate(special_tokens_mask):
            if idx in sentence_entities_positions:
                new_stm.append(0)
            else:
                new_stm.append(1)

        return new_stm


    def get_entities_bert(self):
        
        text_entities_mask = []
        for index, item in enumerate(tqdm.tqdm(self.all_texts, desc = "Extract entities!")):
            item_entities_mask = []
            encoding_entities_ids = []
            # print(self.text_entities[index])
            for _, entity in enumerate(self.text_entities[index]):
                encoding_entities_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity)))
            
            for caption_index, _ in enumerate(item):
                text = self.all_texts[index][caption_index]
                # print(text)
                # print(self.tokenizer.tokenize(text))
                encoding = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_len,
                    return_special_tokens_mask=True,
                )

                item_entities_mask.append(self.exclude_non_entities_bert(encoding["input_ids"], encoding_entities_ids, encoding["special_tokens_mask"]))
            text_entities_mask.append(item_entities_mask)
            # print(item_entities_mask)
            # exit()

        return text_entities_mask 


    def exclude_non_entities_bert(self, inputs, entities_tokens_ids, special_tokens_mask):
        """Exclude non-entity tokens from token masking."""

        sentence_entities_positions = []
        
        for entity_id in entities_tokens_ids:
            for i in range(len(inputs)):
                
                if inputs[i : i + len(entity_id)] == entity_id:
                    
                    sequence_ids = list(range(i, i + len(entity_id)))
                    if random.random() > 0.7:
                        # print(self.tokenizer.decode(entity_id))
                        sentence_entities_positions.extend(sequence_ids)
        self.texts_entity_mask_ratio += len(sentence_entities_positions)/len(special_tokens_mask)
        self.texts_num += 1

        new_stm = []
        for idx, _ in enumerate(special_tokens_mask):
            if idx in sentence_entities_positions:
                new_stm.append(False)
            else:
                new_stm.append(True)
                
        return new_stm