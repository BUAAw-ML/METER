from ..datasets import WitCaptionDataset
from .datamodule_base import BaseDataModule


class WitCaptionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return WitCaptionDataset

    @property
    def dataset_name(self):
        return "wit"
