import json
import numpy as np
from .mp3d_dataset import MP3DDataset
from collections import defaultdict
ERROR_MARGIN = 3.0

class R2RDataset(MP3DDataset):
    def load_data(self, anno_file, max_instr_len=200, debug=False):
        """
        :param anno_file:
        :param max_instr_len:
        :param debug:
        :return:
        """
        with open(str(anno_file), "r") as f:
            data = json.load(f)
        new_data = []
        sample_index = 0

        for i, item in enumerate(data):
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['raw_idx'] = i
                new_item['sample_idx'] = sample_index
                new_item['instr_id'] = 'r2r_{}_{}'.format(item['path_id'], j)

                new_item['instruction'] = instr
                del new_item['instructions']

                if 'instr_encodings' in new_item:
                    new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                    del new_item['instr_encodings']

                if 'new_instructions' in new_item and len(eval(item['new_instructions'])) > j:
                    new_item['fg_instruction'] = eval(item['new_instructions'])[j]
                    new_item['fg_instruction'] = [' '.join(instr) for instr in new_item['fg_instruction']]
                    del new_item['new_instructions']
                    new_item['fg_view'] = item['chunk_view'][j]
                    fg_view = []
                    for idx, index in enumerate(new_item['fg_view']):
                        index_num = index[1] - index[0]
                        fg_view += [idx] * index_num
                    new_item['fg_view'] = fg_view
                    del new_item['chunk_view']

                new_item['data_type'] = 'r2r'
                new_data.append(new_item)
                sample_index += 1

        if debug:
            new_data = new_data[:20]

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
            for x in new_data if len(x['path']) > 1
        }
        return new_data, gt_trajs


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

        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']

            if instr_id not in self.gt_trajs.keys():
                print("instr_id {} not in self.gt_trajs".format(instr_id))
                raise NotImplementedError

            if name == "R2R":
                scan, gt_traj = self.gt_trajs[instr_id]
                traj_scores = self.eval_dis_item(scan, traj, gt_traj)
            else:
                raise NotImplementedError

            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        if name in ['R2R']:
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
        else:
            raise NotImplementedError
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

        for item in results:
            item['instr_id'] = "_".join(item['instr_id'].split("_")[1:])
            item['trajectory'] = [[y, 0, 0] for x in item['trajectory'] for y in x]

        with open(path, 'w') as fout:
            json.dump(results, fout)