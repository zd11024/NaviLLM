import os
import json
import copy
from PIL import Image
from typing import Dict, Any, List
from .image_text_dataset import ImageTextDataset


class COCOCaptionDataset(ImageTextDataset):
    def _load_data(self, config: Dict, data_dir: str):
        path = os.path.join(data_dir, config.coco_caption.DIR, config.coco_caption.SPLIT[self.split])

        self.alldata = []
        with open(path) as f:
            data = json.load(f)
        
            for item in data:
                if self.training:
                    for sent in item["sentences"]:
                        self.alldata.append({
                            "sentid": sent["sentid"],
                            "image": item["filename"].split("_")[-1],
                            "input": "What is the caption of this image?",
                            "label": sent["raw"]+"</s>",
                        })
                else:
                    self.alldata.append({
                        "imgid": item["imgid"],
                        "image": item["filename"].split("_")[-1],
                        "input": "What is the caption of this image?",
                        "refs": [sent["raw"] for sent in item["sentences"]]
                    })

        if self.max_datapoints:
            self.alldata = self.alldata[:self.max_datapoints]
        self.logger.info(f"There are totally {len(self.alldata)} datapoints loaded.")

    def __getitem__(self, index:int) -> Dict[str, Any]:
        item = copy.deepcopy(self.alldata[index])

        # load image
        image_path = os.path.join(self.config.coco_caption.IMAGE_DIR, 'train2017', item["image"])
        if not os.path.exists(image_path):
            image_path = os.path.join(self.config.coco_caption.IMAGE_DIR, 'val2017', item["image"])

        image = Image.open(image_path).convert('RGB')
        
        if self.training:
            data_dict = {
                "sentid": item["sentid"],
                "image": image,
                "input": item["input"],
                "label": item["label"]
            }
        else:
            data_dict = {
                "imgid": item["imgid"],
                "image": image,
                "input": item["input"],
                "refs": item["refs"]
            } 
        
        return data_dict

    def eval_metrics(self, preds: List[Dict[str, Any]], logger, name: str) -> Dict[str, float]:
        refs = {}
        for item in self.alldata:
            refs[item["imgid"]] = item["refs"]
        
        gen = {item['imgid']:item['outputs'] for item in preds}

        from tools.evaluation.bleu import Bleu
        bleu_score = Bleu()

        score, scores = bleu_score.compute_score(refs, gen)

        ret = {}
        for i, s in enumerate(score):
            ret[f"bleu-{i+1}"] = s
        return ret