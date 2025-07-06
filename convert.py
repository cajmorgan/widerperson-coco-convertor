

import json
import os
import shutil
from typing import List, Literal

import PIL
import PIL.Image
from datetime import datetime, timezone


class Converter():

    cwd = ""
    paths = []
    image_id = 1
    annotation_id = 1
    split = "train"

    def __init__(self):
        self.cwd = os.getcwd()
        train_txt_path = f"{self.cwd}/data/WiderPerson/train.txt"
        val_txt_path = f"{self.cwd}/data/WiderPerson/val.txt"

        self.train_image_paths = self.__get_image_paths(train_txt_path)
        self.val_image_paths = self.__get_image_paths(val_txt_path)

        self.annotations_path = f"{self.cwd}/out/annotations/"
        self.train_path = f"{self.cwd}/out/train2017/"
        self.val_path = f"{self.cwd}/out/val2017/"

        # Create files etc
        shutil.rmtree(f"{self.annotations_path}", ignore_errors=True)
        shutil.rmtree(f"{self.train_path}", ignore_errors=True)
        shutil.rmtree(f"{self.val_path}", ignore_errors=True)

        os.mkdir(f"{self.annotations_path}")
        os.mkdir(f"{self.train_path}")
        os.mkdir(f"{self.val_path}")

    def __get_image_paths(self, txt_path):
        with open(txt_path, "r") as f:
            paths = [line.strip() for line in f.readlines()]

        return paths

    def create_annotation_split(self, split: Literal["train", "val"]):
        self.split = split
        if split == "train":
            self.paths = self.train_image_paths
        else:
            self.paths = self.val_image_paths

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

        image_base_path = f"{self.cwd}/out/{split}2017"

        for path in self.paths:
            base_path = f"{self.cwd}/data/WiderPerson"

            image = PIL.Image.open(f"{base_path}/Images/{path}.jpg")
            image.save(f"{image_base_path}/{path}.jpg")
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

        with open(f"{self.annotations_path}/instances_{self.split}2017.json", "w") as f:
            f.write(json.dumps(coco_output))


converter = Converter()
converter.create_annotation_split("train")
converter.create_annotation_split("val")
