import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
# import cv2

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from .bert_model import BertCrossLayer, BertAttention
from . import swin_transformer as swin
from . import heads, objectives, meter_utils
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel
from transformers import AutoModel, T5Model


from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
    RobertaTokenizer,
    AutoTokenizer,
    T5Tokenizer,
)

class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.masking_strategy = config['masking_strategy']

        self.is_clip= (not 'swin' in config['vit'])

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after=config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(
                        pretrained=True, config=self.hparams.config,
                    )

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'],cache_dir="../public")
                elif 'LinkBERT' in config['tokenizer']:
                    AutoModel.from_pretrained(config['tokenizer'],cache_dir="../public")
                elif 't5-small' in config['tokenizer']:
                    T5Model.from_pretrained(config['tokenizer'],cache_dir="../public")
                else:
                    BertModel.from_pretrained(config['tokenizer'],cache_dir="../public")

            torch.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vit_model = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        if 'roberta' in config['tokenizer']:
            print("text_transformer: roberta!")
            self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'],cache_dir="../public")
        elif 'LinkBERT' in config['tokenizer']:
            print("text_transformer: LinkBERT!")
            self.text_transformer = AutoModel.from_pretrained(config['tokenizer'],cache_dir="../public")
        elif 't5-small' in config['tokenizer']:
            print("text_transformer: t5-small!")
            self.text_transformer = T5Model.from_pretrained(config['tokenizer'],cache_dir="../public")
        else:
            print("text_transformer: BERT!")
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'],cache_dir="../public")
            
        # self.Tokenizer = RobertaTokenizer.from_pretrained(config['tokenizer'], cache_dir="../../../public")

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            if "okvqa" in self.hparams.config["datasets"]:
                vs = self.hparams.config["okvqa_label_size"]
            elif "vqav2" in self.hparams.config["datasets"]:
                vs = self.hparams.config["vqav2_label_size"]
            elif "aokvqa" in self.hparams.config["datasets"]:
                vs = self.hparams.config["aokvqa_label_size"]

            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["classifier"] > 0:

            self.classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.classifier.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            # print(ckpt.keys())

            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            
            try:
                self.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print("process state_dict!")
                state_dict.pop("vqa_classifier.3.weight")
                state_dict.pop("vqa_classifier.3.bias")
                self.load_state_dict(state_dict, strict=False)


        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["snli"] > 0:
            self.snli_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 3),
            )
            self.snli_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        meter_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch["text_masks"]

        # print([self.Tokenizer.tokenize(item) for item in batch["text"]])

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)

        device = text_embeds.device
        input_shape = text_masks.size()
        
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)

        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        
        image_embeds = self.vit_model(img)

        # print(image_embeds.shape)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        x, y = text_embeds, image_embeds

        text_self_attention = None
        text_cross_attention = None
        origin_image_cross_attention = None

        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]



            if self.masking_strategy == "entity_masking":

                # print(x1[1].shape) #torch.Size([8, 12, 50, 50])
                # print(x1[2].shape) #torch.Size([8, 12, 50, 325])
                # print(y1[1].shape) #torch.Size([8, 12, 325, 325])
                # print(y1[2].shape) #torch.Size([8, 12, 325, 50])
            
                if text_self_attention == None:  
                    text_self_attention = x1[1].sum(-2).sum(1).unsqueeze(1)
                    text_cross_attention = y1[2].sum(-2).sum(1).unsqueeze(1)
                    origin_image_cross_attention = x1[2].unsqueeze(1)

                else:
                    text_self_attention = torch.cat((text_self_attention, x1[1].sum(-2).sum(1).unsqueeze(1)), 1)
                    text_cross_attention = torch.cat((text_cross_attention, y1[2].sum(-2).sum(1).unsqueeze(1)), 1)
                    origin_image_cross_attention = torch.cat((origin_image_cross_attention, x1[2].unsqueeze(1)), 1)

                
        

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        if self.masking_strategy == "entity_masking":
            # print(text_cross_attention[:,-1,:])
            # print(text_cross_attention.mean(1).shape)

            # ret.update({"text_attention": text_cross_attention[:,-1,:]})
            ret.update({"text_attention": text_cross_attention.mean(1),
                        "origin_image_cross_attention": origin_image_cross_attention[:,0,:,:,:].mean(1),
                        "image": img
                        })
            ret.update({"text_attention": text_cross_attention[:,-1,:]+text_self_attention[:,-1,:]})

            # ret.update({"text_attention": text_cross_attention.mean(1)+text_self_attention.mean(1)})

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        if "classifier" in self.current_tasks:
            ret.update(objectives.compute_classifier(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        # # Only one dataset is supported
        # if self.masking_strategy == "entity_masking":
        #     # self.trainer.datamodule.dms[0].train_dataset.text_attention.update(list(zip(output["text_attention"][0],output["text_attention"][1])))#[output["text_attention"][0][0]] = (output["text_attention"][1][0], output["text_attention"][2][0])
        #     # print(self.trainer.datamodule.dms[0].train_dataset.text_attention[output["text_attention"][0][0]].shape)
        #     print(len(self.trainer.datamodule.dms[0].train_dataset.text_attention))

        #     # exit()
        

        return total_loss

    def training_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        meter_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return meter_utils.set_schedule(self)
