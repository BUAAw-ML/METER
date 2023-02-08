from ..datasets import FOILCOCODataset
from .datamodule_base import BaseDataModule


class FoilCOCODataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return FOILCOCODataset

    @property
    def dataset_name(self):
        return "foil"
