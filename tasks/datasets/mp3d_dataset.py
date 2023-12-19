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

def get_anno_file_path(data_dir, dataset_path, filename):
    if dataset_path.startswith('/'):
        return Path(dataset_path) / filename
    return Path(data_dir) / dataset_path / filename

class MP3DDataset(torch_data.Dataset):
    def __init__(
            self,
            args,
            config,
            training=False,
            logger=None,
            source=None,
    ):
        super().__init__()
        self.config = config
        self.angle_feat_size = self.config.angle_feat_size
        self.logger = logger
        self.training = training
        self.debug = args.debug
        self.source = source

        if self.training:
            self.split = "train"
            self.max_objects = self.config.max_objects
            self.multi_endpoints = True
        else:
            self.split = args.validation_split
            self.max_objects = None
            self.multi_endpoints = False

        self.batch_size = args.batch_size
        self.seed = args.seed
        self.feat_db = None
        self.obj_feat_db = None

        # connectivity graph
        self.connectivity_dir = str(args.data_dir/'connectivity')

        # load mp3d dataset
        msg = self._load_data(config, args.data_dir)
        self.buffered_state_dict = {}

        # simulator
        self.sim = new_simulator(self.connectivity_dir)

        # angle features
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)

        # navigation graph
        self._load_nav_graphs()

        if logger is not None:
            logger.info('[INFO] %s loaded with %d instructions, using splits: %s' % (
                self.__class__.__name__, len(self.alldata), self.split))
            logger.info(msg)
        del self.data

    def init_feat_db(self, feat_db, obj_feat_db=None):
        self.feat_db = feat_db
        self.obj_feat_db = obj_feat_db

    def _load_data(self, config, data_dir):
        self.data = dict()
        self.alldata = []
        msg = ""
        if self.source == "R2R":
            anno_file = get_anno_file_path(data_dir, config.R2R.DIR, config.R2R.SPLIT[self.split])
            self.data['r2r'], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} R2R samples'.format(len(self.data['r2r']))
        elif self.source == "REVERIE":
            anno_file = get_anno_file_path(data_dir, config.REVERIE.DIR, config.REVERIE.SPLIT[self.split])
            bbox_file = get_anno_file_path(data_dir, config.REVERIE.DIR, config.REVERIE.bbox_file)
            obj2vps = self.load_obj2vps(bbox_file)
            self.data['reverie'], self.gt_trajs = self.load_data(anno_file=anno_file, obj2vps=obj2vps, debug=self.debug)
            msg += '\n- Dataset: load {} REVERIE samples'.format(len(self.data['reverie']))
        elif self.source == "CVDN":
            anno_file = get_anno_file_path(data_dir, config.CVDN.DIR, config.CVDN.SPLIT[self.split])
            self.data['cvdn'], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} CVDN samples'.format(len(self.data['cvdn']))
        elif self.source == "SOON":
            anno_file = get_anno_file_path(data_dir, config.SOON.DIR, config.SOON.SPLIT[self.split])
            self.data['soon'], self.gt_trajs = self.load_data(anno_file=anno_file, debug=self.debug)
            msg += '\n- Dataset: load {} SOON samples'.format(len(self.data['soon']))
        elif self.source == "R2R_AUG":
            anno_file = get_anno_file_path(data_dir, config.R2R_AUG.DIR, config.R2R_AUG.SPLIT[self.split])
            self.data["r2r_aug"], _ = self.load_data(anno_file=anno_file, debug=self.debug)
        elif self.source == "REVERIE_AUG":
            anno_file = get_anno_file_path(data_dir, config.REVERIE_AUG.DIR, config.REVERIE_AUG.SPLIT[self.split])
            bbox_file = get_anno_file_path(data_dir, config.REVERIE.DIR, config.REVERIE.bbox_file)
            obj2vps = self.load_obj2vps(bbox_file)
            self.data["reverie_aug"], _ = self.load_data(anno_file=anno_file, obj2vps=obj2vps, debug=self.debug)
        elif self.source == "EQA":
            anno_file = get_anno_file_path(data_dir, config.EQA.DIR, config.EQA.SPLIT[self.split])
            self.data['eqa'], self.gt_trajs = self.load_data(anno_file=anno_file, split=self.split, debug=self.debug)
        else:
            print("Dataset Source: {}".format(self.source))
            raise NotImplementedError

        for key, value in self.data.items():
            self.alldata += value

        msg += '\n- Dataset: load {} split: {} samples in total'.format(self.split, len(self.alldata))
        self.scans = set([x['scan'] for x in self.alldata])
        msg += '\n- Dataset: load {} split: {} scans in total'.format(self.split, len(self.scans))

        return msg

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        # print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, index):
        item = copy.deepcopy(self.alldata[index])
        item = self.preprocess_item(item)
        data_type = item['data_type']
        scan = item['scan']
        instr_id = item['instr_id']

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
    
    def preprocess_item(self, item):
        return item

    @staticmethod
    def collate_batch(batch_list, _unused=False):
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
    
    def get_object_info(self, item):
        raise NotImplementedError

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
                'instruction': item['instruction'],
                # 'instr_encoding': item['instr_encoding'],
                'gt_path': item['path'],
                'path_id': item['path_id'],
            }
            if 'fg_instruction' in item:
                ob.update({
                    'fg_instruction': item['fg_instruction'],
                    'fg_view': item['fg_view'],
                })
            if self.obj_feat_db is not None:
                obj_info = self.get_object_info(item, state)
                ob.update(obj_info)
                ob['distance'] = 0
            else:
                # RL reward. The negative distance between the state and the final state
                # There are multiple gt end viewpoints on REVERIE. 
                if False: # ob['instr_id'] in self.gt_trajs:
                    ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
                else:
                    ob['distance'] = 0
            obs.append(ob)
        return obs

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)

        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)

                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId,  # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }

            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = feature[ix]
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id