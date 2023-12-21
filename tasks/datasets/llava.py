import os
import json
import copy
import torch
import logging
import numpy as np
from collections import defaultdict
from PIL import Image
from typing import List, Dict, Any, Union
from .base_dataset import BaseDataset


class LLaVADataset(BaseDataset):
    name = 'llava'

    def __init__(
        self,
        args,
        config: Dict,
        training: bool=False,
        logger: logging.Logger=None,
        source: str=None,
    ):
        super().__init__()
        self.config = config
        self.training = training
        self.logger = logger
        self.source = source

        if training:
            self.split = 'train'
        else:
            self.split = args.validation_split

        self.batch_size = args.batch_size
        self.feat_db = None
        self.obj_feat_db = None
        self.max_datapoints = args.max_datapoints

        self._load_data(config, args.data_dir)


    def init_feat_db(self, feat_db, obj_feat_db=None):
        self.feat_db = feat_db
        self.obj_feat_db = obj_feat_db


    def _load_data(self, config: Dict, data_dir: str):

        if self.source == "LLaVA":
            role_mapping = {
                "human": "USER",
                "gpt": "ASSISTANT",
            }
            seps = [" ", "</s>"]

            path = os.path.join(data_dir, config.LLaVA.DIR, config.LLaVA.SPLIT[self.split])
            with open(path) as f:
                data = json.load(f)
                self.alldata = []

                for item in data:
                    image = item["image"]
                    conversations = item["conversations"]
                    prompt = ""
                    assert len(conversations)==2, "The round of conversation must be 2."
                    for i in range(0, len(conversations)-1, 2):
                        assert conversations[i]["from"]=="human", f"The {i}-th utterance must come from human!"
                        assert conversations[i+1]["from"]=="gpt", f"The {i+1}-th utterance must come from agent!"
                        self.alldata.append({
                            "id": item["id"],
                            "turn_id": i//2,
                            "image_id": item["image"].split(".")[0],
                            # "input": role_mapping[conversations[i]["from"]] + ": " + conversations[i]["value"] + seps[0] + role_mapping[conversations[i+1]["from"]] + ": ",
                            # "label": conversations[i+1]["value"] + seps[1]  # </s> indicates the end of generation.
                            "question": conversations[i]["value"].replace("<image>", "").strip(),
                            "answers": [conversations[i+1]["value"]]
                        })

                        # prompt += role_mapping[conversations[i]["from"]] + ": " + conversations[i]["value"] + seps[0] + role_mapping[conversations[i+1]["from"]] + ": " + conversations[i+1]["value"] + seps[1]

            if self.max_datapoints:
                self.alldata = self.alldata[:self.max_datapoints]
            self.logger.info(f"There are totally {len(self.alldata)} datapoints loaded.")

        else:
            raise NotImplementedError
    

    def __len__(self) -> int:
        return len(self.alldata)

    
    def __getitem__(self, index:int) -> Dict[str, Any]:
        item = copy.deepcopy(self.alldata[index])

        # load image
        features = self.feat_db.get_image_feature(item["image_id"])
        features = torch.from_numpy(np.stack(features)).unsqueeze(0)
        
        data_dict = {
            "id": item["id"],
            "image_id": item["image_id"],
            "question": item["question"],
            "answers": item["answers"],
            "data_type": "llava",
            "features": features,
        }
        
        return data_dict


    @staticmethod
    def collate_batch(batch_list: List[Dict], _unused: bool=False) -> Dict[str, Union[List[Any], torch.Tensor]]:
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['NotImplemented']:
                    ret[key] = torch.stack(val, 0)
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
