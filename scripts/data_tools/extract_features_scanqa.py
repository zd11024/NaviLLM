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
import h5py
import argparse
from more_itertools import batched
import psutil


@ray.remote(num_gpus=1)
def process_features(proc_id, out_queue, scenevp_list, args):
    print(f"Start process {proc_id}, there are {len(scenevp_list)} datapoints")
    sys.path.append("EVA/EVA-CLIP/rei")
    from eva_clip import create_model_and_transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load visual encoder
    model, _, transform = create_model_and_transforms(args.model_name, args.pretrained, force_custom_clip=True)
    visual_encoder = model.visual.to(device)
    visual_encoder.eval()

    # for scene_id, image_id in scenevp_list:
    for i, batch in enumerate(batched(scenevp_list, args.batch_size)):
        # Loop all discretized views from this location
        images = []
        for item in batch:
            image = Image.open(item["path"])
            images.append(image)

        vision_x = [transform(image).unsqueeze(0).to(device) for image in images]
        vision_x = torch.cat(vision_x, dim=0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            outs = visual_encoder.forward_features(vision_x)
        outs = outs.data.cpu().numpy()

        for i, item in enumerate(batch):
            out_queue.put((item["scene_id"], item["image_id"], outs[i], []))
        
        if i%1000==0:
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"Memory used by current process: {memory_info.rss / (1024 * 1024):.2f} MB")

    out_queue.put(None)

@ray.remote
def write_features(out_queue, total, num_workers, args):

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
                scene_id, image_id, fts, logits = res
                key = '%s_%s' % (scene_id, image_id)
                if False:
                    data = np.hstack([fts, logits])
                else:
                    data = fts # shape=(36, 1408)
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['sceneId'] = scene_id
                outf[key].attrs['imageId'] = image_id

                num_finished_vps += 1
                if num_finished_vps % 20000 == 0:
                    print("num_finished_vps: ", num_finished_vps)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print("data shape: ", data.shape)
                progress_bar.update(num_finished_vps)

    progress_bar.finish()

import time
def main(args):

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)


    image_list = []
    for scene_id in os.listdir(args.image_dir):
        if scene_id.endswith(".py") or scene_id.endswith(".txt"):
                continue
        for filename in os.listdir(os.path.join(args.image_dir, scene_id, "color")):
            image_list.append({
                "path": os.path.join(args.image_dir, scene_id, "color", filename),
                "scene_id": scene_id,
                "image_id": filename.split('.')[0] 
            })
    print("Loaded %d viewpoints" % len(image_list))
    print(image_list[0])

    scenevp_list = image_list
    num_workers = min(args.num_workers, len(scenevp_list))
    num_data_per_worker = len(scenevp_list) // num_workers

    ray.init()
    out_queue = Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = process_features.remote(proc_id, out_queue, scenevp_list[sidx: eidx], args)
        processes.append(process)

    process = write_features.remote(out_queue, len(scenevp_list), num_workers, args)
    processes.append(process)

    ray.get(processes)
    ray.shutdown()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EVA02-CLIP-L-14-336")
    parser.add_argument("--pretrained", type=str, default="data/models/EVA02_CLIP_L_336_psz14_s6B.pt", help="the path of EVA-CLIP")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_dir', type=str, default="data/ScanQA/frames_square/", help='the original ScanQA dataset with RGB frames')
    parser.add_argument("--output_file", type=str, default="data/eva_features/scanqa_EVA02-CLIP-L-14-336.hdf5", help="the path of output features")
    args = parser.parse_args()

    main(args)
