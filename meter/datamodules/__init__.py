from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .snli_datamodule import SNLIDataModule
from .okvqa_datamodule import OKVQADataModule
from .wit_caption_datamodule import WitCaptionDataModule
from .data_utils import DataCollatorForEntityLanguageModeling

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "snli": SNLIDataModule,
    "vqav2": VQAv2DataModule,
    "okvqa": OKVQADataModule,
    "wit": WitCaptionDataModule,
}

save_special_tokens_mask = {}