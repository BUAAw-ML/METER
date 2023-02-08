from .base_dataset import BaseDataset


class FOILCOCODataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["foilcoco_train"]
        elif split == "val":
            names = ["foilcoco_val"]
        elif split == "test":
            names = ["foilcoco_val"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption", remove_duplicate=False)

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, info_index = self.index_mapper[index]
        foil = self.table["info"][index][info_index]['foil'].as_py()

        return {
            "image": image_tensor,
            "text": text,
            "foil": foil
        } 
