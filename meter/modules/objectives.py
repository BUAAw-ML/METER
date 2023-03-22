import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
import numpy as np

from .dist_utils import all_gather

from PIL import Image
from torchvision import utils,transforms
# from visualize import visualize_grid_attention_v2

from copy import deepcopy


def probe(img, text_self_attention = None, text_cross_attention = None,  image_self_attention = None, image_cross_attention = None):

    # text_self_attention #torch.Size([8, 12, 50, 50])
    # image_cross_attention #torch.Size([8, 12, 50, 325])
    # image_self_attention #torch.Size([8, 12, 325, 325])
    #text_cross_attention  #torch.Size([8, 12, 325, 50])


    ii = img[0].cpu().permute(1,2,0)*torch.tensor([0.26862954, 0.26130258, 0.27577711]) + torch.tensor([0.48145466, 0.4578275, 0.40821073])
    image=transforms.ToPILImage()(ii.permute(2,0,1))
    image.save('cat2.png')

    image_cross_attention = image_cross_attention.mean(1)
    grid = (image_cross_attention.shape[1] - 1) ** 0.5

    grid = int(grid)

    jj=image_cross_attention[:,1:].reshape(image_cross_attention.shape[0],grid,grid)
    
    jj = jj.cpu().numpy()

    jj=np.repeat(jj,16,axis=-1)
    jj=np.repeat(jj,16,axis=-2)

    # print((text_cross_attention.mean(1)).shape)
    # print((torch.max(text_cross_attention.mean(1),-1)))

    # mask = cv2.resize(attention_mask, (img_h, img_w))
    # print(mask.max())

    normed_mask = jj[0] / jj[0].max()
    normed_mask = (normed_mask * 255).astype('uint8')

    img_path='cat2.png'
    save_path="test"
    # attention_mask = np.random.randn(14, 14)
    visualize_grid_attention_v2(img_path,
                                save_path=save_path,
                                attention_mask=normed_mask,
                                save_image=True,
                                save_original_image=True,
                                quality=100)
    exit()



    # print((text_cross_attention.mean(1) > 0.5).shape)
    # print((text_cross_attention.mean(1) > 0.5).sum(-1))
    # print((image_cross_attention.mean(-2) > 0.5).sum(-1))
    exit()

def compute_mlm(pl_module, batch):

    if "text_entities_masks_info" in batch:

        # print(batch["text"][0])#
        # print(pl_module.trainer.datamodule.dms[0].train_dataset.tokenizer.tokenize(batch["text"][0])[:50])
        # print(batch["text_entities_masks_info"].sum(-1).gt(0))
        
        infer = pl_module.infer(batch, mask_text=True, mask_image=False)
        
        entities_attention = infer["text_attention"] * batch["text_entities_masks_info"].sum(-1).gt(0)  #( ~ batch["text_entities_masks"].bool())

        threshold = torch.max(torch.cat((entities_attention.sort(1, descending=True)[0][:,2].unsqueeze(-1), torch.tensor([0.00001] * batch["text_entities_masks_info"].shape[0]).cuda(entities_attention.device).unsqueeze(-1)),dim=-1),-1, keepdim=True)[0]

        target_entities_position = entities_attention.ge(threshold).unsqueeze(-2) #entities_attention.max(1, keepdim=True)[0]
        
        text_entities_masks = torch.matmul(target_entities_position.float(), batch["text_entities_masks_info"].float()).squeeze(1).bool()
        # print(text_entities_masks[0].unsqueeze(0))

        # image_cross_attention2entities = text_entities_masks[0].unsqueeze(0).unsqueeze(-1) * infer["origin_image_cross_attention"].detach()
        # probe(infer["image"], image_cross_attention=image_cross_attention2entities)
        text_ids_mlm = deepcopy(batch["text_ids_mlm"])
        batch["text_labels_mlm"].masked_fill_(~text_entities_masks, value=-100) #Mask tensor can take 0 and 1 values only
        batch["text_ids_mlm"].masked_fill_(text_entities_masks, value=batch["mask_token"])
        
        # print(batch["text_labels_mlm"])
        # print(batch["text_ids_mlm"])
        # exit()

    infer = pl_module.infer(batch, mask_text=True, mask_image=False)

    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    if "text_entities_masks_info" in batch:

        # print(mlm_logits.shape)
        # print(mlm_labels.shape)

        mlm_labels2= deepcopy(mlm_labels)
        mlm_labels2[mlm_labels==-100] = 0

        a = F.log_softmax(mlm_logits, dim=-1)
        b = F.one_hot(mlm_labels2, pl_module.hparams.config["vocab_size"])
        c = (-a * b)
        c = c.sum(-1)
        c[mlm_labels==-100] = 0
        # print(c)
        # print(c.shape)
        target_entities_position = c.ge(c.sort(-1, descending=True)[0][:,0].unsqueeze(-1)).unsqueeze(-2)
        
        text_entities_masks = torch.matmul(target_entities_position.float(), batch["text_entities_masks_info"].float()).squeeze(1).bool()
        batch["text_labels_mlm"].masked_fill_(~text_entities_masks, value=-100) #Mask tensor can take 0 and 1 values only
        text_ids_mlm.masked_fill_(text_entities_masks, value=batch["mask_token"])
        batch["text_ids_mlm"] = text_ids_mlm
        # print(batch["text_labels_mlm"])
        # print(batch["text_ids_mlm"])

        # exit()
        # mlm_loss = (c.sum(-1) / (c != 0).sum(-1)).mean() 
            # exit()

        infer = pl_module.infer(batch, mask_text=True, mask_image=False)

        mlm_logits = pl_module.mlm_score(infer["text_feats"])
        mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )



    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret

def compute_snli(pl_module, batch):
    infer = pl_module.infer(
        batch, mask_text=False, mask_image=False, 
    )
    snli_logits = pl_module.snli_classifier(infer["cls_feats"])

    snli_labels = batch["labels"]
    snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()
    snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))

    ret = {
        "snli_loss": snli_loss,
        "snli_logits": snli_logits,
        "snli_labels": snli_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_loss")(ret["snli_loss"])
        acc = getattr(pl_module, f"{phase}_snli_accuracy")(
            ret["snli_logits"], ret["snli_labels"]
        )
        pl_module.log(f"snli/{phase}/loss", loss)
        pl_module.log(f"snli/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_accuracy")(
                ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
            )
            pl_module.log(f"snli/dev/loss", dev_loss)
            pl_module.log(f"snli/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_accuracy")(
                ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
            )
            pl_module.log(f"snli/test/loss", test_loss)
            pl_module.log(f"snli/test/accuracy", test_acc)

    return ret

def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])

    if "okvqa" in pl_module.hparams.config["datasets"]:
        vs = pl_module.hparams.config["okvqa_label_size"]
    elif "vqav2" in pl_module.hparams.config["datasets"]:
        vs = pl_module.hparams.config["vqav2_label_size"]
    elif "aokvqa" in pl_module.hparams.config["datasets"]:
        vs = pl_module.hparams.config["aokvqa_label_size"]

    vqa_targets = torch.zeros(
        len(vqa_logits), vs
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret

def compute_classifier(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    logits = pl_module.classifier(infer["cls_feats"])

    labels = torch.tensor(batch['foil']).to(pl_module.device).long()

    loss = F.cross_entropy(
        logits.view(-1, 2),
        labels.view(-1)
    )

    ret = {
        "loss": loss,
        "logits": logits,
        "labels": labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_classifier_loss")(ret["loss"])
    acc = getattr(pl_module, f"{phase}_classifier_accuracy")(
        ret["logits"], ret["labels"]
    )
    pl_module.log(f"classifier/{phase}/loss", loss)
    pl_module.log(f"classifier/{phase}/accuracy", acc)

    return ret



def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    #TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        img=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except:
        # id2answer = (
        #     pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
        #     if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
        #     else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        # )
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["okvqa_trainval"].id2answer
            if "okvqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["okvqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]

        vqa_answers = []
        for labels, scores in zip(batch["vqa_labels"], batch["vqa_scores"]):
            vqa_answers.append([(id2answer[label],score)  for label,score in zip(labels, scores)])

        questions = batch["text"]
        qids = batch["qid"]
        img_id = batch["img_id"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True, "questions": questions, "vqa_answers": vqa_answers, "img_id": img_id}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    questions, vqa_answers, img_id = list(), list(), list()
    gqa = False

    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']
        if gqa:
            questions += out["questions"]
            vqa_answers += out["vqa_answers"]
            img_id += out["img_id"] 

    rets = list()
    
    if gqa:
        for qid, pred, questions, vqa_answers, img_id in zip(qids, preds, questions, vqa_answers, img_id):
            rets.append({"img_id": img_id, "questionId": qid, "questions": questions, "prediction": pred, "vqa_answers": vqa_answers})
    else:
        for qid, pred in zip(qids, preds):
            rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        jsons.sort(key=lambda x: (x['img_id']))
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
