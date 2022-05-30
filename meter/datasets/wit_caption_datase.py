from .base_dataset import BaseDataset
import io
from PIL import Image

class WitCaptionDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["coco_caption_karpathy_train"]
        elif split == "val":
            names = ["coco_caption_karpathy_test"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")


    def __getitem__(self, index):
        suite = self.get_suite(index)

        return suite
