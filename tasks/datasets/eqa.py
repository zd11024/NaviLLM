import json
import copy
from pathlib import Path
import torch.utils.data as torch_data
import torch
import math
from collections import defaultdict
import numpy as np
import networkx as nx
from .mp3d_envs import (
    EnvBatch, new_simulator, angle_feature,
    get_all_point_angle_feature, load_nav_graphs,
)
ERROR_MARGIN = 3.0
from tools.evaluation.bleu import Bleu
from tools.evaluation.rouge import Rouge
from tools.evaluation.cider import Cider
from .mp3d_dataset import MP3DDataset, get_anno_file_path


class EQADataset(MP3DDataset):
    def __init__(
            self,
            args,
            config,
            training=False,
            logger=None,
            source=None,
    ):
        super().__init__(args, config, training, logger, source)
        # answer_vocab
        filename = get_anno_file_path(args.data_dir, config.EQA.DIR, config.EQA.ANSWER_VOCAB)
        with open(filename) as f:
            self.answer_vocab = json.load(f)


    def init_feat_db(self, feat_db, obj_feat_db=None):
        self.feat_db = feat_db
        self.obj_feat_db = obj_feat_db


    def load_data(self, anno_file, max_instr_len=200, split='train', debug=False):
        """
        :param anno_file:
        :param max_instr_len:
        :param debug:
        :return:
        """
        with open(str(anno_file), "r") as f:
            data = json.load(f)
        new_data = []

        for i, item in enumerate(data):
            new_item = dict(item)
            new_item['raw_idx'] = item['sample_idx']
            new_item['instr_id'] = 'eqa_{}_{}'.format(item['sample_idx'], i)
            new_item['path_id'] = item['sample_idx']
            new_item['data_type'] = 'eqa'
            new_item['heading'] = 0.0
            new_data.append(new_item)

        if debug:
            new_data = new_data[:20]

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }
        return new_data, gt_trajs


    def get_obs(self, items, env, data_type=None):
        obs = []

        for i, (feature, state) in enumerate(env.getStates()):
            item = items[i]
            base_view_id = state.viewIndex

            if feature is None:
                feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            ob = {
                'instr_id': item['instr_id'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'candidate': candidate,
                'navigableLocations': state.navigableLocations,
                'instruction': item['question']['question_text'],
                'answer': item['question']['answer_text'],
                # 'instr_encoding': item['instr_encoding'],
                'gt_path': item['path'],
                'path_id': item['path_id'],
            }
            if False: # ob['instr_id'] in self.gt_trajs:
                ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            else:
                ob['distance'] = 0
            obs.append(ob)
        return obs

    ########################### Evalidation ###########################
    def eval_metrics(self, preds, logger, name):
        """
        Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]
        :param preds:
        :param logger:
        :param name:
        :return:
        """
        logger.info('eval %d predictions' % (len(preds)))
        metrics = defaultdict(list)
        all_pred_ans = {}
        all_gt_ans = {}
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            pred_ans = item['pred_answer']
            gt_ans = item['gt_answer']
            all_pred_ans[instr_id] = pred_ans
            all_gt_ans[instr_id] = [gt_ans]

            if instr_id not in self.gt_trajs.keys():
                print("instr_id {} not in self.gt_trajs".format(instr_id))
                raise NotImplementedError

            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self.eval_dis_item(scan, traj, gt_traj)
          
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
        }

        # bleu_score = Bleu()
        # score, scores = bleu_score.compute_score(all_gt_ans, all_pred_ans)
        # for i, s in enumerate(score):
        #     avg_metrics[f"bleu-{i+1}"] = s * 100        

        # rouge_score = Rouge()
        # score, compute_score = rouge_score.compute_score(all_gt_ans, all_pred_ans)
        # avg_metrics["rouge"] = score * 100

        # cider_score = Cider()
        # score, compute_score = cider_score.compute_score(all_gt_ans, all_pred_ans)
        # avg_metrics["cider"] = score * 100
        n_correct = 0
        for pred in preds: 
            if pred['pred_answer'] in all_gt_ans[pred["instr_id"]]: 
                n_correct += 1
        avg_metrics["exact_match"] = n_correct / len(preds) * 100

        n_oracle_correct = 0
        for pred in preds:
            if pred['oracle_pred_answer'] in all_gt_ans[pred['instr_id']]:
                n_oracle_correct += 1
        avg_metrics["oracle_exact_match"] = n_oracle_correct / len(preds) * 100

        return avg_metrics, metrics

    def eval_dis_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self.get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        return scores

    def save_json(self, results, path, item_metrics=None):
        if item_metrics is not None:
            for k in item_metrics:
                for item, v in zip(results, item_metrics[k]):
                    item[k] = v

        with open(path, 'w') as fout:
            json.dump(results, fout)