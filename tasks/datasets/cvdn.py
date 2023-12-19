import json
import copy
import numpy as np
from .mp3d_dataset import MP3DDataset
from collections import defaultdict
from .mp3d_envs import (
    EnvBatch,
)
ERROR_MARGIN = 3.0


class CVDNDataset(MP3DDataset):
    name = "cvdn"
    
    def __init__(
            self,
            args,
            config,
            training=False,
            logger=None,
            source=None,
    ):
        super().__init__(args, config, training, logger, source)

        if args.max_datapoints:
            self.alldata = self.alldata[:args.max_datapoints]

        for idx, item in enumerate(self.alldata):
            if args.path_type=="trusted_path" and "end_panos" in item and item['path'][-1] not in item['end_panos']:
                dist_start_to_end = None
                goal_vp = None
                for end_vp in item['end_panos']:
                    d = self.shortest_paths[item['scan']][item['start_pano']['pano']][end_vp]
                    if dist_start_to_end is None or len(d) < len(dist_start_to_end):
                        dist_start_to_end = d
                        goal_vp = end_vp
                item['path'] = self.shortest_paths[item['scan']][item['start_pano']['pano']][goal_vp]

    def load_data(self, anno_file, debug=False, path_type='trusted_path'):
        with open(str(anno_file), "r") as f:
            data = json.load(f)
        new_data = []
        sample_idx = 0
        for i, item in enumerate(data):
            new_item = dict(item)
            new_item['heading'] = None  # item['start_pano']['heading']
            if path_type == 'trusted_path':
                if 'planner_path' in item:
                    new_item['path'] = item['planner_path']
                else:
                    new_item['path'] = [item['start_pano']['pano']]
            else:
                raise NotImplementedError

            if len(item['dialog_history']) == 0:
                new_item['instruction'] = "The goal room contains a {target}.\n".format(target=item['target'])
            else:
                new_item['instruction'] = "The goal room contains a {target}.\n".format(target=item['target'])
                sentences = []
                for turn in item['dialog_history']:
                    if turn['message'][-1] == '?' or turn['message'][-1] == '.':
                        msg = turn['message']
                    else:
                        msg = turn['message'] + "."
                    if turn['role'] == 'navigator':
                        sentences.append("Question: " + msg + "\n")
                    elif turn['role'] == 'oracle':
                        sentences.append("Answer: " + msg + "\n")
                    else:
                        raise NotImplementedError
                sentences = "".join(sentences)
                new_item['instruction'] += sentences
            if new_item['instruction'][-1] == '\n':
                new_item['instruction'] = new_item['instruction'][:-1]
            new_item['path_id'] = item['inst_idx']
            new_item['raw_idx'] = None
            new_item['instr_encoding'] = None
            new_item['data_type'] = 'cvdn'
            new_item['sample_idx'] = sample_idx
            new_item['instr_id'] = 'cvdn_{}_{}'.format(sample_idx, new_item['path_id'])

            new_data.append(new_item)
            sample_idx += 1
        if debug:
            new_data = new_data[:20]

        gt_trajs = {}
        for x in new_data:
            gt_trajs[x['instr_id']] = x

        return new_data, gt_trajs


    def __getitem__(self, index):
        item = copy.deepcopy(self.alldata[index])
        data_type = item['data_type']
        scan = item['scan']
        instr_id = item['instr_id']
        item['heading'] = item['start_pano']['heading']
        scanIds = [scan]
        viewpointIds = [item['path'][0]]
        headings = [item['heading']]

        # check length of instruction
        max_len = 128
        if len(item['instruction'].split()) > max_len:
            self.alldata[index]['instruction'] = " ".join(item['instruction'].split()[:max_len])
            item['instruction'] = " ".join(item['instruction'].split()[:max_len])

        env = EnvBatch(connectivity_dir=self.connectivity_dir, batch_size=1)
        env.newEpisodes(scanIds, viewpointIds, headings)
        observations = self.get_obs(items=[item], env=env, data_type=data_type)[0]

        data_dict = {
            'sample_idx': index,
            'instr_id': instr_id,
            'observations': observations,
            'env': env,
            'item': item,
            'data_type': data_type,
        }

        return data_dict

    def eval_metrics(self, preds, logger, name):
        logger.info('eval %d predictions' % (len(preds)))
        metrics = defaultdict(list)

        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']

            if instr_id not in self.gt_trajs.keys():
                print("instr_id {} not in self.gt_trajs".format(instr_id))
                raise NotImplementedError

            traj = sum(traj, [])
            gt_item = self.gt_trajs[instr_id]
            traj_scores = self.eval_cvdn(gt_item['scan'], traj, gt_item)

            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        if name in ['CVDN']:
            num_successes = len([i for i in metrics['nav_errors'] if i < ERROR_MARGIN])
            oracle_successes = len([i for i in metrics['oracle_errors'] if i < ERROR_MARGIN])
            oracle_plan_successes = len([i for i in metrics['oracle_plan_errors'] if i < ERROR_MARGIN])

            avg_metrics = {
                'lengths': np.average(metrics['trajectory_lengths']),
                'nav_error': np.average(metrics['nav_errors']),
                'oracle_sr': float(oracle_successes) / float(len(metrics['oracle_errors'])) * 100,
                'sr': float(num_successes) / float(len(metrics['nav_errors'])) * 100,
                'spl': np.mean(metrics['spl']) * 100,
                'oracle path_success_rate': float(oracle_plan_successes) / float(
                    len(metrics['oracle_plan_errors'])) * 100,
                'dist_to_end_reduction': sum(metrics['dist_to_end_reductions']) / float(
                    len(metrics['dist_to_end_reductions']))
            }
        else:
            raise NotImplementedError
        return avg_metrics, metrics

    def eval_cvdn(self, scan, path, gt_item):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        shortest_distances = self.shortest_distances[scan]

        start = gt_item['path'][0]
        assert start == path[0], 'Result trajectories should include the start position'
        goal = gt_item['path'][-1]
        planner_goal = gt_item['planner_path'][-1]  # for calculating oracle planner success (e.g., passed over desc goal?)
        final_position = path[-1]  # 预测
        nearest_position = self.get_nearest(shortest_distances, goal, path)
        nearest_planner_position = self.get_nearest(shortest_distances, planner_goal, path)
        dist_to_end_start = None
        dist_to_end_end = None
        for end_pano in gt_item['end_panos']:
            d = shortest_distances[start][end_pano]  # distance from start -> goal
            if dist_to_end_start is None or d < dist_to_end_start:
                dist_to_end_start = d
            d = shortest_distances[final_position][end_pano]  # dis from pred_end -> goal
            if dist_to_end_end is None or d < dist_to_end_end:
                dist_to_end_end = d

        scores = dict()
        scores['nav_errors'] = shortest_distances[final_position][goal]
        scores['oracle_errors'] = shortest_distances[nearest_position][goal]
        scores['oracle_plan_errors'] = shortest_distances[nearest_planner_position][planner_goal]
        scores['dist_to_end_reductions'] = dist_to_end_start - dist_to_end_end
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            if prev != curr:
                try:
                    self.graphs[gt_item['scan']][prev][curr]
                except KeyError as err:
                    print(err)
            distance += shortest_distances[prev][curr]
            prev = curr
        scores['trajectory_lengths'] = distance
        scores['success'] = float(scores['nav_errors'] < ERROR_MARGIN)
        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip( gt_item['path'][:-1],  gt_item['path'][1:])])
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['shortest_path_lengths'] = shortest_distances[start][goal]
        return scores
    
    def save_json(self, results, path, item_metrics=None):
        for item in results:
            item['trajectory'] = [[y, 0, 0] for x in item['trajectory'] for y in x]
            item['instr_idx'] = item['inst_idx'] = item['instr_id'].split('_')[-1]
            item['instr_idx'] = item['inst_idx'] = int(item['inst_idx'])
        
        with open(path, 'w') as f:
            json.dump(results, f)