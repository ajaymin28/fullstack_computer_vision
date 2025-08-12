from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import os

class plant_disease_ds(Dataset):
    """
    Loads plant disease as torch dataset
    """

    def __init__(self, data_root, subset="train", transforms=None):
        valid_subsets = ["train", "test", "validation"]
        assert subset in valid_subsets

        self.subset = subset
        self.data_root = data_root
        self.transforms = transforms

        ## setup data indexing
        self.data = []
        self.data_classes = []
        
        ## class str-id maps
        self.cls_to_ids = {}
        self.ids_to_cls = {}

        self.subset_dir = os.path.join(data_root, subset)

        for root, dirs, files in os.walk(self.subset_dir):
            for dir in dirs:
                if dir not in self.cls_to_ids:
                    keys_len = len(self.cls_to_ids.keys())
                    self.ids_to_cls[keys_len] = dir
                    self.cls_to_ids[dir] = keys_len
                
                cls_dir = os.path.join(root, dir)


                ## Define data pattern *.jpg, *.jpeg, *.png
                filePattern = os.path.join(cls_dir, "*.jpg")
                current_class_data = glob(filePattern)

                # appends string class for each image
                for i in range(len(current_class_data)):
                    self.data_classes.append(dir)

                self.data += current_class_data



    def __str__(self):

        ## class wise counter 
        self.cls_counter = {}

        for cur_cls in self.data_classes:
            if cur_cls in self.cls_counter:
                self.cls_counter[cur_cls] +=1 
            else:
                self.cls_counter[cur_cls] = 1 

        return f"""
        cls_counter: {self.cls_counter}
        Total samples: {len(self)}
        cls_to_ids: {self.cls_to_ids}
        ids_to_cls: {self.ids_to_cls}
        """

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):


        img = Image.open(self.data[index])
        if self.transforms is not None:
            img = self.transforms(img)

        # str class
        str_cls = self.data_classes[index]

        # id class
        id_cls = self.cls_to_ids[str_cls]

        return {
            "img": img,
            "str_cls": str_cls,
            "id_cls": id_cls
        }

        # return img, str_cls, id_cls