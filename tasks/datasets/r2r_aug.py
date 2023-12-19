import json
import numpy as np
from .r2r import R2RDataset
from collections import defaultdict
from transformers import AutoTokenizer

class R2RAugDataset(R2RDataset):
    name = "r2r_aug"

    def load_data(self, anno_file, max_instr_len=200, debug=False):
        """
        :param anno_file:
        :param max_instr_len:
        :param debug:
        :return:
        """
        if str(anno_file).endswith(".json"):
            return super().load_data(anno_file, max_instr_len=max_instr_len, debug=debug)

        with open(str(anno_file), "r") as f:
            data = []
            for i, line in enumerate(f.readlines()):
                if debug and i==20:
                    break
                data.append(json.loads(line.strip()))
        new_data = []
        sample_idx = 0
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        for i, item in enumerate(data):
            new_item = dict(item)
            new_item["raw_idx"] = i
            new_item["sample_idx"] = sample_idx
            new_item['data_type'] = 'r2r_aug'
            new_item["path_id"] = None
            new_item["heading"] = item.get("heading", 0)
            new_item["instruction"] = tokenizer.decode(new_item['instr_encoding'], skip_special_tokens=True)
            new_data.append(new_item)
            sample_idx += 1

        if debug:
            new_data = new_data[:20]

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }
        return new_data, gt_trajs

