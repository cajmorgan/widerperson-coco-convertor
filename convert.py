

import os
from typing import List, Literal

import PIL
import PIL.Image
from datetime import datetime, timezone


class Converter():

    cwd = ""
    paths = []
    image_id = 1
    annotation_id = 1

    def __init__(self, split: Literal["train", "val"]):
        self.cwd = os.getcwd()
        train_txt_path = f"{self.cwd}/data/WiderPerson/train.txt"
        val_txt_path = f"{self.cwd}/data/WiderPerson/val.txt"

        train_image_paths = self.__get_image_paths(train_txt_path)
        val_image_paths = self.__get_image_paths(val_txt_path)

        if split == "train":
            self.paths = train_image_paths
        else:
            self.paths = val_image_paths

    def __get_image_paths(self, txt_path):
        with open(txt_path, "r") as f:
            paths = [line.strip() for line in f.readlines()]

        return paths

    def create_annotation_split(self):

        coco_output = {
            "info": {
                "year": str(datetime.now().year),
                "version": "11",
                "description": "some dataset",
                "contributor": "",
                "url": "http://shuoyang1213.me/WIDERFACE/",
                "date_created": datetime.now(timezone.utc).isoformat()
            },
            "license": [{
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0"
            }],
            "images": [],
            "annotations": [],
            # Note that we specify only one category as every annotation is assumed to a person
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "none"
                }
            ]
        }

        for path in self.paths:
            base_path = f"{self.cwd}/data/WiderPerson"

            image = PIL.Image.open(f"{base_path}/Images/{path}.jpg")
            width, height = image.size

            image_json = {
                "id": self.image_id,
                "file_name": f"{path}.jpg",
                "width": width,
                "height": height
            }

            coco_output["images"].append(image_json)
            with open(f"{base_path}/Annotations/{path}.jpg.txt", "r") as f:
                annotations = [line.strip() for line in f.readlines()]
                annotations = annotations[1:]  # skip count

            for annot in annotations:
                _, x1, y1, x2, y2 = annot.split(" ")

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                bbox_width = x2 - x1
                bbox_height = y2 - y1

                coco_output["annotations"].append({
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": 1,  # one class only
                    "bbox": [x1, y1, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })

                self.annotation_id += 1
            self.image_id += 1

        return coco_output


converter = Converter("train")

out = converter.create_annotation_split()

print(out)
