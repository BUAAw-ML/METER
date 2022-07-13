from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import re


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import random
import torch

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase



@dataclass
class DataCollatorForEntityLanguageModeling(DataCollatorForLanguageModeling):
    """Implements Entity masking so that only entites within the 'entities.txt' file will
    be masked.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_probability: float = 0.8
    mlm: bool = True
    mlm_probability: float = 1 #0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    entities_file_path: str = "entities.txt"

    

    def __post_init__(self):
        super().__post_init__()
 
        # Exclude non entity tokens from Token Masking
        # with open("entities.txt", "r") as f:
        #     entities = f.read().split("\n")

        # entities_tokens = [self.tokenizer.tokenize(' '.join(re.split('_',entity.split("\"")[1].split("(")[0]))) for entity in entities]

        # self.entities_tokens_ids = []
        # for tokenized_entity in entities_tokens:
        #     entity_id = self.tokenizer.convert_tokens_to_ids(tokenized_entity)
        #     # print(tokenized_entity)

        #     # decod = self.tokenizer.decode(entity_id)
        #     # if decod == 'A':
        #     #     print(tokenized_entity)
        #     #     print(len(tokenized_entity)) #1
        #     #     print(len(tokenized_entity[0])) #1
        #     if len(tokenized_entity) == 0:
        #         continue 
        #     elif len(tokenized_entity) >1:
        #         self.entities_tokens_ids.append(entity_id)

        #     elif len(tokenized_entity[0]) > 1 and tokenized_entity[0] != 'Are':
        #         self.entities_tokens_ids.append(entity_id)
        #     else:
        #          print(tokenized_entity)
                
        # print(f'The number of entities: {len(self.entities_tokens_ids)}!')


        # self.count = 0


    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random,
        10% original.
        """
        # print(self.mlm_probability)

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        # (with probability `self.mlm_probability`)

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()


        # special_tokens_mask = self.exclude_non_entities(
        #     inputs, self.entities_tokens_ids, special_tokens_mask
        # )

        # CLS and other tokens are excluded from the token masking
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_probability)).bool()
            & masked_indices
        )

        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
   
    # def exclude_non_entities(self, inputs, entities_tokens_ids, special_tokens_mask):
    #     """Exclude non-entity tokens from token masking."""

    #     global save_special_tokens_mask

    #     all_entities_positions = []

    #     # get exception ids
    #     for instance_ids in inputs.tolist():
    #         sentence_entities_positions = []

    #         special_tokens_mask_identifier = '-'.join(map(str,instance_ids))
    #         if 1:# special_tokens_mask_identifier not in self.save_special_tokens_mask.keys():

    #             for entity_id in entities_tokens_ids:

    #                 for i in range(len(instance_ids)):
                        
    #                     if instance_ids[i : i + len(entity_id)] == entity_id:
    #                         # print(self.tokenizer.decode(entity_id))
    #                         sequence_ids = list(range(i, i + len(entity_id)))
    #                         sentence_entities_positions.extend(sequence_ids)
    #         else:
    #             print("1111111111111111111111")
    #             sentence_entities_positions.extend([-1])
    #         sentence_entities_positions.extend([special_tokens_mask_identifier])
    #         # print(f"sentence_entities_positions!{sentence_entities_positions}")

    #         all_entities_positions.append(sentence_entities_positions)
        
    #     # print(all_entities_positions)
    #     new_special_tokens_mask = []

    #     # manipulate special tokens mask list by excluding
    #     for stm_idx, stm in enumerate(special_tokens_mask):
    #         if 0:#all_entities_positions[stm_idx][0] == -1:
    #             print("22222222222222222222222")
    #             new_special_tokens_mask.append(save_special_tokens_mask[all_entities_positions[stm_idx][-1]])
    #         else:
    #             new_stm = []
    #             for idx, _ in enumerate(stm):
    #                 if idx in all_entities_positions[stm_idx]:
    #                     new_stm.append(False)
    #                 else:
    #                     new_stm.append(True)
    #             new_special_tokens_mask.append(new_stm)
                
    #             save_special_tokens_mask[all_entities_positions[stm_idx][-1]] = new_stm
    #     # self.count += 8
    #     print(len(save_special_tokens_mask.keys()))
    #     # print(f"new_special_tokens_mask!{new_special_tokens_mask}")
    #     return torch.tensor(new_special_tokens_mask)




# # Copyright 2020 The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple, Union

# import random
# import torch

# from transformers.file_utils import PaddingStrategy
# from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

# def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
#     """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
#     # Tensorize if necessary.
#     if isinstance(examples[0], (list, tuple)):
#         examples = [torch.tensor(e, dtype=torch.long) for e in examples]

#     # Check if padding is necessary.
#     length_of_first = examples[0].size(0)
#     are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
#     if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
#         return torch.stack(examples, dim=0)

#     # If yes, check if we have a `pad_token`.
#     if tokenizer._pad_token is None:
#         raise ValueError(
#             "You are attempting to pad samples but the tokenizer you are using"
#             f" ({tokenizer.__class__.__name__}) does not have a pad token."
#         )

#     # Creating the full tensor and filling it with our data.
#     max_length = max(x.size(0) for x in examples)
#     if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
#         max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
#     result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
#     for i, example in enumerate(examples):
#         if tokenizer.padding_side == "right":
#             result[i, : example.shape[0]] = example
#         else:
#             result[i, -example.shape[0] :] = example
#     return result


# def tolist(x: Union[List[Any], torch.Tensor]):
#     return x.tolist() if isinstance(x, torch.Tensor) else x

# @dataclass
# class DataCollatorForLanguageModeling:
#     """
#     Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
#     are not all of the same length.
#     Args:
#         tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
#             The tokenizer used for encoding the data.
#         mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
#             Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
#             inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
#             non-masked tokens and the value to predict for the masked token.
#         mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
#             The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.
#     .. note::
#         For best performance, this data collator should be used with a dataset having items that are dictionaries or
#         BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
#         :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
#         argument :obj:`return_special_tokens_mask=True`.
#     """

#     tokenizer: PreTrainedTokenizerBase
#     mlm: bool = True
#     mlm_probability: float = 0.15
#     pad_to_multiple_of: Optional[int] = None

#     def __post_init__(self):
#         if self.mlm and self.tokenizer.mask_token is None:
#             raise ValueError(
#                 "This tokenizer does not have a mask token which is necessary for masked language modeling. "
#                 "You should pass `mlm=False` to train on causal language modeling instead."
#             )

#     def __call__(
#         self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
#     ) -> Dict[str, torch.Tensor]:
#         # Handle dict or lists with proper padding and conversion to tensor.
#         if isinstance(examples[0], (dict, BatchEncoding)):
#             batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
#         else:
#             batch = {"input_ids": _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

#         # If special token mask has been preprocessed, pop it from the dict.
#         special_tokens_mask = batch.pop("special_tokens_mask", None)
#         if self.mlm:
#             batch["input_ids"], batch["labels"] = self.mask_tokens(
#                 batch["input_ids"], special_tokens_mask=special_tokens_mask
#             )
#         else:
#             labels = batch["input_ids"].clone()
#             if self.tokenizer.pad_token_id is not None:
#                 labels[labels == self.tokenizer.pad_token_id] = -100
#             batch["labels"] = labels
#         return batch

#     def mask_tokens(
#         self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
#         """
#         labels = inputs.clone()
#         # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
#         probability_matrix = torch.full(labels.shape, self.mlm_probability)
#         if special_tokens_mask is None:
#             special_tokens_mask = [
#                 self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#             ]
#             special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
#         else:
#             special_tokens_mask = special_tokens_mask.bool()

#         probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#         masked_indices = torch.bernoulli(probability_matrix).bool()
#         labels[~masked_indices] = -100  # We only compute loss on masked tokens

#         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]

#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#         return inputs, labels




# class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
#     """
#     Data collator used for language modeling.

#     - collates batches of tensors, honoring their tokenizer's pad_token
#     - preprocesses batches for masked language modeling
#     """

#     def __call__(
#         self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
#     ) -> Dict[str, torch.Tensor]:
#         if isinstance(examples[0], (dict, BatchEncoding)):
#             input_ids = [e["input_ids"] for e in examples]
#         else:
#             input_ids = examples
#             examples = [{"input_ids": e} for e in examples]

#         batch_input = _collate_batch(input_ids, self.tokenizer)

#         mask_labels = []
#         for e in examples:
#             ref_tokens = []
#             for id in tolist(e["input_ids"]):
#                 token = self.tokenizer._convert_id_to_token(id)
#                 ref_tokens.append(token)


#             mask_labels.append(self._whole_word_mask(ref_tokens))
#         batch_mask = _collate_batch(mask_labels, self.tokenizer)
#         inputs, labels = self.mask_tokens(batch_input, batch_mask)
#         return {"input_ids": inputs, "labels": labels}

#     def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
#         """
#         Get 0/1 labels for masked tokens with whole word mask proxy
#         """

#         cand_indexes = []
#         for (i, token) in enumerate(input_tokens):
#             if token == "[CLS]" or token == "[SEP]":
#                 continue

#             if len(cand_indexes) >= 1 and token.startswith("##"):
#                 cand_indexes[-1].append(i)
#             else:
#                 cand_indexes.append([i])

#         random.shuffle(cand_indexes)
#         # 这里的mask 15%的也是字，而不是15%的词
#         # 只不过mask的时候是根据词来mask，但是统计是否达到了15%是根据字来统计
#         num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
#         masked_lms = []
#         covered_indexes = set()
#         for index_set in cand_indexes:
#             if len(masked_lms) >= num_to_predict:
#                 break
#             # If adding a whole-word mask would exceed the maximum number of
#             # predictions, then just skip this candidate.
#             if len(masked_lms) + len(index_set) > num_to_predict:
#                 continue
#             is_any_index_covered = False
#             for index in index_set:
#                 if index in covered_indexes:
#                     is_any_index_covered = True
#                     break
#             if is_any_index_covered:
#                 continue
#             for index in index_set:
#                 covered_indexes.add(index)  # 统计是否达到了mask 15%的字的比例，而不是词
#                 masked_lms.append(index)

#         assert len(covered_indexes) == len(masked_lms)
#         mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
#         return mask_labels

#     def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
#         'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
#         """

#         if self.tokenizer.mask_token is None:
#             raise ValueError(
#                 "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
#             )
#         labels = inputs.clone()
#         # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

#         # 这里的mask_labels就相当于已经实现了15%的字MASK之后的masked_indices(DataCollatorForLanguageModeling类中
#         # 根据伯努利分布生成的)
#         probability_matrix = mask_labels

#         special_tokens_mask = [
#             self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#         ]
#         probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
#         if self.tokenizer._pad_token is not None:
#             padding_mask = labels.eq(self.tokenizer.pad_token_id)
#             probability_matrix.masked_fill_(padding_mask, value=0.0)

#         masked_indices = probability_matrix.bool()
#         labels[~masked_indices] = -100  # We only compute loss on masked tokens

#         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]

#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#         return inputs, labels
