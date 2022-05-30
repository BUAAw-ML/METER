from ..datasets import ConceptualCaptionDataset
from .datamodule_base import BaseDataModule


class WitCaptionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return wit_caption_datase

    @property
    def dataset_name(self):
        return "wit"
