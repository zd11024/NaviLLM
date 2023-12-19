import json
import copy
import numpy as np
from .mp3d_dataset import MP3DDataset
from collections import defaultdict

class REVERIEDataset(MP3DDataset):
    name = "reverie"

    def __init__(
        self,
        args,
        config,
        training=False,
        logger=None,
        source=None,
    ):
        super().__init__(args, config, training, logger, source)
        self.multi_startpoints = False
        self.multi_endpoints = args.multi_endpoints

    def preprocess_item(self, item):
        if self.split!="train" or "end_vps" not in item or (not self.multi_startpoints and not self.multi_endpoints):
            return item

        start_vp = item["path"][0]
        end_vp = item["path"][-1]

        if self.multi_startpoints:
            cand_vps = []
            for cvp, cpath in self.shortest_paths[item['scan']][end_vps[i]].items():
                if len(cpath) >= 4 and len(cpath) <= 7:
                    cand_vps.append(cvp)
            if len(cand_vps) > 0:
                start_vp = cand_vps[np.random.randint(len(cand_vps))]

        if self.multi_endpoints:
            end_vp = item["end_vps"][np.random.randint(len(item["end_vps"]))]

        item = copy.deepcopy(item)
        item["path"] = self.shortest_paths[item["scan"]][start_vp][end_vp]
        return item

    def load_data(self, anno_file, obj2vps, debug=False):
        with open(str(anno_file), "r") as f:
            data = json.load(f)

        new_data = []
        sample_index = 0
        for i, item in enumerate(data):
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)

                if 'objId' in item:
                    new_item['instr_id'] = '%s_%s_%s_%d' % ('reverie', str(item['path_id']), str(item['objId']), j)
                else:
                    new_item['path_id'] = item['id']
                    new_item['instr_id'] = '%s_%s_%d' % ('reverie', item['id'], j)
                    new_item['objId'] = None

                new_item['sample_idx'] = sample_index
                new_item['instruction'] = instr
                del new_item['instructions']
                new_item['data_type'] = 'reverie'

                new_item['raw_idx'] = None
                new_item['instr_encoding'] = None

                if 'objId' in item and item['objId'] is not None:
                    new_item['end_vps'] = obj2vps['%s_%s'%(item['scan'], item['objId'])]

                new_data.append(new_item)
                sample_index += 1
        if debug:
            new_data = new_data[:20]

        gt_trajs = {
            x['instr_id']: (x['scan'], x['path'], x['objId']) \
            for x in new_data if 'objId' in x and x['objId'] is not None
        }

        return new_data, gt_trajs


    def load_obj2vps(self, bbox_file):
        obj2vps = {}
        bbox_data = json.load(open(bbox_file))
        for scanvp, value in bbox_data.items():
            scan, vp = scanvp.split('_')
            # for all visible objects at that viewpoint
            for objid, objinfo in value.items():
                if objinfo['visible_pos']:
                    # if such object not already in the dict
                    obj2vps.setdefault(scan+'_'+objid, [])
                    obj2vps[scan+'_'+objid].append(vp)
        self.obj2vps = obj2vps
        return obj2vps
    
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
            pred_objid = item.get('pred_objid', None)
            scan, gt_traj, gt_objid = self.gt_trajs[instr_id]
            traj_scores = self.eval_dis_item(scan, traj, pred_objid, gt_traj, gt_objid)

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
            'rgs': np.mean(metrics['rgs']) * 100,
            'rgspl': np.mean(metrics['rgspl']) * 100
        }

        return avg_metrics, metrics

    def eval_dis_item(self, scan, pred_path, pred_objid, gt_path, gt_objid):
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

        # navigation: success is to arrive to a viewpoint where the object is visible
        goal_viewpoints = set(self.obj2vps['%s_%s'%(scan, str(gt_objid))])
        assert len(goal_viewpoints) > 0, '%s_%s'%(scan, str(gt_objid))

        scores['success'] = float(path[-1] in goal_viewpoints)
        scores['oracle_success'] = float(any(x in goal_viewpoints for x in path))
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        scores['rgs'] = str(pred_objid) == str(gt_objid)
        scores['rgspl'] = scores['rgs'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        return scores
    
    def get_object_info(self, item, state):
        # objects
        obj_img_fts, obj_ang_fts, obj_box_fts, obj_ids = \
            self.obj_feat_db.get_object_feature(
                state.scanId, state.location.viewpointId,
                state.heading, state.elevation, self.angle_feat_size,
                max_objects=self.max_objects
            )
        
        gt_end_vps = item.get('end_vps', []) 
        
        gt_obj_id = None
        vp = state.location.viewpointId
        if vp in gt_end_vps:
            gt_obj_id = item['objId']

        return {
            'obj_img_fts': obj_img_fts,
            'obj_ang_fts': obj_ang_fts,
            'obj_box_fts': obj_box_fts,
            'obj_ids': obj_ids,
            'gt_end_vps': gt_end_vps,
            'gt_obj_id': gt_obj_id,
        }
    
    def save_json(self, results, path, item_metrics=None):    
        if item_metrics is not None:
            for k in item_metrics:
                for item, v in zip(results, item_metrics[k]):
                    item[k] = v
    
        for item in results:
            item['instr_id'] = "_".join(item['instr_id'].split("_")[1:])
            item['trajectory'] = [[y, 0, 0] for x in item['trajectory'] for y in x]
            item['predObjId'] = int(item['pred_objid']) if item['pred_objid'] is not None else 0
        
        with open(path, 'w') as f:
            json.dump(results, f)

