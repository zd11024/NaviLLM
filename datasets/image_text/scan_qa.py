import os
import json
import copy
from PIL import Image
from typing import Dict, Any, List, Tuple
from .image_text_dataset import ImageTextDataset
import random
import torch
import numpy as np
from tools.evaluation.bleu import Bleu
from tools.evaluation.rouge import Rouge
from tools.evaluation.cider import Cider
from tools.evaluation.meteor import Meteor

class ScanQADataset(ImageTextDataset):
    def _load_data(self, config: Dict, data_dir: str):
        if config.ScanQA.DIR.startswith("/"):
            path = os.path.join(config.ScanQA.DIR, config.ScanQA.SPLIT[self.split])
        else:
            path = os.path.join(data_dir, config.ScanQA.DIR, config.ScanQA.SPLIT[self.split])

        self.alldata = []
        with open(path) as f:
            data = json.load(f)
            for item in data:
                for ann in item["annotation"]:
                    self.alldata.append({
                        "question_id": ann["question_id"],
                        "question": ann["question"],
                        "answers": [ans.lower() for ans in ann["answers"]],
                        "image_info": item["image_info"],
                        "scene_id": item["scene_id"]
                    })

        if self.max_datapoints:
            self.alldata = self.alldata[:self.max_datapoints]
        self.logger.info(f"There are totally {len(self.alldata)} datapoints loaded.")

    def __getitem__(self, index:int) -> Dict[str, Any]:
        item = copy.deepcopy(self.alldata[index])

        sampled_images = random.sample(item["image_info"], min(36, len(item["image_info"])))
        features = []
        for d in sampled_images:
            fts = self.feat_db.get_image_feature(item["scene_id"], d["image_id"])
            features.append(fts)
        features = torch.from_numpy(np.stack(features))

        data_dict = {
            "scene_id": item["scene_id"],
            "question_id": item["question_id"],
            "question": item["question"],
            "answers": item["answers"],
            "features": features,
            "data_type": "scan_qa"
        }
        return data_dict

    def eval_metrics(self, preds: List[Dict[str, Any]], logger, name: str) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        ret = {}
        if self.split=='test':
            return ret

        refs = {}
        for item in self.alldata:
            refs[item["question_id"]] = item["answers"]     
        gen = {item['question_id']:item['generated_sentences'] for item in preds}

        bleu_score = Bleu()
        score, scores = bleu_score.compute_score(refs, gen)
        for i, s in enumerate(score):
            ret[f"bleu-{i+1}"] = s * 100        

        rouge_score = Rouge()
        score, compute_score = rouge_score.compute_score(refs, gen)
        ret["rouge"] = score * 100

        cider_score = Cider()
        score, compute_score = cider_score.compute_score(refs, gen)
        ret["cider"] = score * 100

        meteor_score = Meteor()
        score, compute_score = meteor_score.compute_score(refs, gen)
        ret["meteor"] = score * 100

        n_correct = 0
        metrics = {"exact_match": []}
        for pred in preds:
            if pred['generated_sentences'][0] in refs[pred["question_id"]]:
                n_correct += 1
                metrics["exact_match"].append(1.)
            else:
                metrics["exact_match"].append(0.)
        ret["exact_match"] = n_correct / len(preds) * 100
        
        return ret, metrics
    
    def save_json(self, results, path, item_metrics=None):
        for item in results:
            item['answer_top10'] = item['generated_sentences']
            item['pred_bbox'] = []
            del item['generated_sentences']
        
        with open(path, 'w') as f:
            json.dump(results, f)