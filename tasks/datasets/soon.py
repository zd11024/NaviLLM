import json
import copy
import jsonlines
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from .mp3d_dataset import MP3DDataset
from collections import defaultdict
from .mp3d_envs import (
    EnvBatch,
)
import math
ERROR_MARGIN = 3.0


class SOONDataset(MP3DDataset):
    name = "soon"

    def __init__(
            self,
            args,
            config,
            training=False,
            logger=None,
            source=None,
    ):
        super().__init__(args, config, training, logger, source)
    
    def load_data(self, anno_file, debug=False):
        data = []
        with jsonlines.open(str(anno_file), 'r') as f:
            for item in f:
                item['end_image_ids'] = [x['image_id'] for x in item['bboxes']]
                item['image_id_to_obj_label'] = {x['image_id']: x.get('pseudo_label', None) for x in item['bboxes']}
                new_bboxes = {}
                for bbox in item['bboxes']:
                    new_bboxes[bbox['image_id']] = bbox
                item['bboxes'] = new_bboxes
                data.append(item)

        new_data = []
        sample_index = 0
        for i, item in enumerate(data):
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = copy.deepcopy(item)
                # soon: idx-path_idx-instr_idx
                new_item['instr_id'] = "soon_{}_{}_{}".format(i, item['path_id'], j)
                new_item['instruction'] = instr['full']
                new_item['instr_encoding'] = item['instr_encodings'][j]['full'][:100]
                del new_item['instructions']
                del new_item['instr_encodings']

                new_item['sample_idx'] = sample_index
                new_item['raw_idx'] = None
                new_item['heading'] = 0.0
                new_item['data_type'] = 'soon'
                new_data.append(new_item)
                sample_index += 1
        if debug:
            new_data = new_data[:20]

        gt_trajs = self._get_gt_trajs(new_data)
        return new_data, gt_trajs

    def __getitem__(self, index):
        item = copy.deepcopy(self.alldata[index])
        data_type = item['data_type']
        scan = item['scan']
        instr_id = item['instr_id']

        if self.training:
            item['heading'] = np.random.rand() * np.pi * 2
            batch = [item]
            start_vps = [x['path'][0] for x in batch]
            end_vps = [x['path'][-1] for x in batch]
            if self.multi_endpoints:
                for i, one_item in enumerate(batch):
                    end_vp = one_item['end_image_ids'][np.random.randint(len(one_item['end_image_ids']))]
                    end_vps[i] = end_vp
            for i, one_item in enumerate(batch):
                one_item['path'] = self.shortest_paths[one_item['scan']][start_vps[i]][end_vps[i]]
            item = batch[0]
        else:
            item['heading'] = 1.52
        item['elevation'] = 0

        scanIds = [scan]
        viewpointIds = [item['path'][0]]
        headings = [item['heading']]

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

    def _get_gt_trajs(self, data):
        # for evaluation
        gt_trajs = {
            x['instr_id']: copy.deepcopy(x) for x in data if 'bboxes' in x 
        }
        # normalize
        for path_id, value in gt_trajs.items():
            new_bboxes = {}
            for vp, bbox in value['bboxes'].items():
                new_bbox = copy.deepcopy(bbox)
                new_bbox['heading'] = new_bbox['target']['center']['heading'] / (2 * math.pi)
                new_bbox['elevation'] = (new_bbox['target']['center']['elevation'] + math.pi) / (2 * math.pi)
                new_bbox['target']['left_top']['heading'] = new_bbox['target']['left_top']['heading'] / (2 * math.pi)
                new_bbox['target']['left_top']['elevation'] = (new_bbox['target']['left_top']['elevation'] + math.pi) / (2 * math.pi)
                new_bbox['target']['right_bottom']['heading'] = new_bbox['target']['right_bottom']['heading'] / (2 * math.pi)
                new_bbox['target']['right_bottom']['elevation'] = (new_bbox['target']['right_bottom']['elevation'] + math.pi) / (2 * math.pi)
                new_bbox['target']['left_bottom']['heading'] = new_bbox['target']['left_bottom']['heading'] / (2 * math.pi)
                new_bbox['target']['left_bottom']['elevation'] = (new_bbox['target']['left_bottom']['elevation'] + math.pi) / (2 * math.pi)
                new_bbox['target']['right_top']['heading'] = new_bbox['target']['right_top']['heading'] / (2 * math.pi)
                new_bbox['target']['right_top']['elevation'] = (new_bbox['target']['right_top']['elevation'] + math.pi) / (2 * math.pi)
                new_bboxes[vp] = new_bbox
            gt_trajs[path_id]['bboxes'] = new_bboxes
        return gt_trajs

    def eval_metrics(self, preds, logger, name):
        logger.info('eval %d predictions' % (len(preds)))
        metrics = defaultdict(list)

        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']

            gt_item = self.gt_trajs[instr_id]
            obj_heading = item['pred_obj_direction'][0] if item['pred_obj_direction'] is not None else None
            obj_elevation = item['pred_obj_direction'][1] if item['pred_obj_direction'] is not None else None
            traj_scores = self.eval_soon_item(traj, gt_item, obj_heading, obj_elevation)

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
            'det_sr': np.mean(metrics['det_success']) * 100,
            'det_spl': np.mean(metrics['det_spl']) * 100,
            
        }
        return avg_metrics, metrics

    def eval_soon_item(self, traj, gt_item, obj_heading, obj_elevation):
        scores = {}
        scan = gt_item['scan']

        shortest_distances = self.shortest_distances[scan]

        gt_path = gt_item['path']
        gt_bboxes = gt_item['bboxes']
        start_vp = gt_path[0]
        goal_vp = gt_path[-1]

        pred_path = traj
        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        # follow the original evaluation
        nearest_position = self.get_nearest(shortest_distances, goal_vp, path)

        if path[-1] in gt_bboxes and (obj_heading is not None) and (obj_elevation is not None):
            gt_bbox = gt_bboxes[path[-1]]
            scores['heading_error'] = math.fabs(gt_bbox['heading'] - obj_heading)
            scores['elevation_error'] = math.fabs(gt_bbox['elevation'] - obj_elevation)
            scores['point_det_error'] = math.hypot(
                gt_bbox['heading'] - obj_heading, gt_bbox['elevation'] - obj_elevation)
            
            # TODO: there might be a bug due to radians angle as it is a circle
            obj_point = Point(obj_heading, obj_elevation)
            gt_poly = Polygon([(gt_bbox['target']['left_top']['heading'], gt_bbox['target']['left_top']['elevation']),
                               (gt_bbox['target']['right_top']['heading'], gt_bbox['target']['right_top']['elevation']),
                               (gt_bbox['target']['right_bottom']['heading'], gt_bbox['target']['right_bottom']['elevation']),
                               (gt_bbox['target']['left_bottom']['heading'], gt_bbox['target']['left_bottom']['elevation'])])

            if gt_poly.contains(obj_point):
                scores['det_success'] = True
            else:
                scores['det_success'] = False
            
        else:
            scores['det_success'] = False

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        # navigation: success is navigation error < 3m
        scores['nav_error'] = shortest_distances[path[-1]][goal_vp]
        # nearest_position = self._get_nearest(shortest_distances, goal_vp, path)
        scores['oracle_error'] = shortest_distances[nearest_position][goal_vp]
        scores['success'] = scores['nav_error'] < 3.
        scores['oracle_success'] = scores['oracle_error'] < 3.

        scores['goal_progress'] = shortest_distances[start_vp][goal_vp] - \
                                  shortest_distances[path[-1]][goal_vp]

        # gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        gt_lengths = shortest_distances[gt_path[0]][goal_vp]

        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['det_spl'] = scores['det_success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        return scores
    
    def get_object_info(self, item, state):
        # objects
        obj_img_fts, obj_ang_fts, obj_box_fts, obj_directions, obj_ids = \
            self.obj_feat_db.get_object_feature(
                state.scanId, state.location.viewpointId,
                state.heading, state.elevation, self.angle_feat_size,
                max_objects=self.max_objects
            )
        
        gt_end_vps = item.get('end_image_ids', [])
        
        gt_obj_id = None
        vp = state.location.viewpointId
        if vp in gt_end_vps:
            pseudo_label = item['image_id_to_obj_label'][vp]
            if pseudo_label is not None:
                gt_obj_id = pseudo_label['obj_id']

        return {
            ### SOON Object
            'obj_img_fts': obj_img_fts,
            'obj_ang_fts': obj_ang_fts,
            'obj_box_fts': obj_box_fts,
            'obj_directions': obj_directions,
            'obj_ids': obj_ids,
            'gt_end_vps': gt_end_vps,
            'gt_obj_id': gt_obj_id,
        }

    def save_json(self, results, path, item_metrics=None):
        new_results = []
        for item in results:
            instr_id = int(item['instr_id'].split("_")[2].split('-')[0])
            new_item = {
                'instr_id': instr_id,
                'trajectory':[{
                    "path": [[y, 0, 0] for x in item['trajectory'] for y in x],
                    "obj_heading": [item['pred_obj_direction'][0] if item['pred_obj_direction'] is not None else 0],
                    "obj_elevation": [item['pred_obj_direction'][1] if item['pred_obj_direction'] is not None else 0]
                }]
            }
            new_results.append(new_item)
        
        with open(path, 'w') as f:
            json.dump(new_results, f)