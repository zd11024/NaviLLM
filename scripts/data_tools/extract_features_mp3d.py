import os
import sys
import numpy as np
import json
import collections
import cv2
import torch
import torch.nn as nn
import ray
from ray.util.queue import Queue
from torchvision import transforms
from PIL import Image
import math
# sys.path.append(mp3d_path)    # please add the simulator path to yout python path. 
import MatterSim
import h5py
import argparse


def build_simulator(connectivity_dir, scan_dir):
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

@ray.remote(num_gpus=1)
def process_features(proc_id, out_queue, scanvp_list, args):
    sys.path.append("EVA/EVA-CLIP/rei")
    from eva_clip import create_model_and_transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # load visual encoder
    model, _, transform = create_model_and_transforms(args.model_name, args.pretrained, force_custom_clip=True)
    visual_encoder = model.visual.to(device)
    visual_encoder.eval()

    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
        for ix in range(36):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        vision_x = [transform(image).unsqueeze(0).to(device) for image in images]
        vision_x = torch.cat(vision_x, dim=0)

        fts = []
        for k in range(0, len(images), args.batch_size):
            input_img = vision_x[k: k + args.batch_size]
            with torch.no_grad(), torch.cuda.amp.autocast():
                outs = visual_encoder.forward_features(input_img)
            outs = outs.data.cpu().numpy()
            fts.append(outs)
        fts = np.concatenate(fts, 0)

        out_queue.put((scan_id, viewpoint_id, fts, []))

    out_queue.put(None)

@ray.remote
def write_features(out_queue, total, num_workers, args):
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    num_finished_workers = 0
    num_finished_vps = 0

    from progressbar import ProgressBar
    progress_bar = ProgressBar(total)
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts, logits = res
                key = '%s_%s' % (scan_id, viewpoint_id)
                if False:
                    data = np.hstack([fts, logits])
                else:
                    data = fts # shape=(36, 1408)
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                if num_finished_vps % 20 == 0:
                    print("num_finished_vps: ",num_finished_vps)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print("data shape: ", data.shape)
                progress_bar.update(num_finished_vps)

    progress_bar.finish()

import time
def main(args):

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    scanvp_list = viewpoint_ids
    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    ray.init()
    out_queue = Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = process_features.remote(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        processes.append(process)

    process = write_features.remote(out_queue, len(scanvp_list), num_workers, args)
    processes.append(process)

    ray.get(processes)
    ray.shutdown()


if __name__ == '__main__':

    scan_data_dir = '/mnt/petrelfs/zhaolin/vln/nav/features/mp3d'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EVA02-CLIP-L-14-336")
    parser.add_argument("--pretrained", type=str, default="data/models/EVA02_CLIP_L_336_psz14_s6B.pt", help='the path of pre-trained model')
    parser.add_argument('--connectivity_dir', default='data/connectivity', help='the path of connectivity')
    parser.add_argument('--scan_dir', default=scan_data_dir)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--output_file", type=str, default="data/eva_features/mp3d_EVA02-CLIP-L-14-336.hdf5", help="the path of output features")
    args = parser.parse_args()

    main(args)

