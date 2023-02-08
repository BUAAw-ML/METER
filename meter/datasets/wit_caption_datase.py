from .base_dataset import BaseDataset
import io
from PIL import Image

class WitCaptionDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":   #["wit0", "wit1", "wit2", "wit3", "wit4", "wit5", "wit6", "wit7", "wit8", "wit9"] #["wit0_small_Cnet"]
            names =["wit0_small", "wit1_small", "wit2_small", "wit3_small", "wit4_small", "wit5_small", "wit6_small", "wit7_small", "wit8_small", "wit9_small"] #, ,"coco_caption_karpathy_test"
        elif split == "val":
            names = ["coco_caption_karpathy_test"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]



        super().__init__(*args, **kwargs, names=names, text_column_name="caption") #


    def __getitem__(self, index):
        suite = self.get_suite(index)

        return suite
