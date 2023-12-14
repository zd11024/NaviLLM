import copy
from tools.common_utils import get_dist_info
import torch
import torch.distributed as dist
from .agent import NavigationAgent
from typing import List, Dict, Tuple, Union, Iterator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from .image_text import (
    ImageTextDataset, 
    COCOCaptionDataset,
    ScanQADataset
)
from .mp3d import (
    R2RDataset, 
    CVDNDataset, 
    SOONDataset,
    EQADataset,
    REVERIEDataset,
    R2RAugDataset,
    REVERIEAugDataset
)

def create_dataloaders(args, config, logger, training, device, feat_db=None, obj_feat_db=None, stage="multi"):
    if training==False and stage=='pretrain':
        return None, None

    dataset_cfg = copy.deepcopy(config.Dataset)
    dataset_cfg.update(
        config.Pretrain if stage=="pretrain" else config.Multi
    )
    dataset_cfg.update(config.Feature)

    dataloaders = {}
    agents = {}
    if args.test_datasets is not None and not training:
        dataset_list = args.test_datasets
    else:
        dataset_list = copy.deepcopy(dataset_cfg.SOURCE)
    for k, task_name in enumerate(dataset_list):

        if task_name == "R2R":
            dataset = R2RDataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name == 'REVERIE':
            dataset = REVERIEDataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name == 'CVDN':
            dataset = CVDNDataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name == 'SOON':
            dataset = SOONDataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name == 'EQA':
            dataset = EQADataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name in ["ScanQA"]:
            dataset = ScanQADataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name in ["LLaVA"]:
            # Note: LLaVa is only available for training
            if task_name=="LLaVA" and not training:
                continue
            dataset = ImageTextDataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name in ["coco_caption"]:
            dataset = COCOCaptionDataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name == "R2R_AUG":
            dataset = R2RAugDataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        elif task_name == "REVERIE_AUG":
            dataset = REVERIEAugDataset(args, dataset_cfg, training=training, logger=logger, source=task_name)
        else:
            raise NotImplementedError(task_name)

        # assign feature database
        if task_name in ["R2R", "REVERIE", "CVDN", "SOON", "EQA", "R2R_AUG", "REVERIE_AUG"]:
            task_feat_db = feat_db['mp3d']
        elif task_name in ["ScanQA"]:
            task_feat_db = feat_db['scan_qa']
        elif task_name in ["LLaVA"]:
            task_feat_db = feat_db["coco"]
        else:
            raise NotImplementedError
        
        # assign object database
        if args.enable_og:
            if task_name in ["REVERIE", "REVERIE_AUG"]:
                task_obj_feat_db = obj_feat_db['reverie']
            elif task_name == "SOON":
                task_obj_feat_db = obj_feat_db['soon']
            else:
                task_obj_feat_db = None
        else:
            task_obj_feat_db = None

        dataset.init_feat_db(feat_db=task_feat_db, obj_feat_db=task_obj_feat_db)


        logger.info(f"{task_name}: {len(dataset)} samples loaded")

        task_loader, pre_epoch = build_dataloader(
            dataset, distributed=args.distributed,
            training=training, batch_size=args.batch_size if training else args.val_batch_size, num_workers=args.workers
        )

        if training:
            ratio = dataset_cfg.Ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device=device)

        if task_name in ["LLaVA", "coco_caption", "ScanQA"]:
                agents[task_name] = None
        else:
            agents[task_name] = NavigationAgent(args, dataset.shortest_distances, dataset.shortest_paths)

    if training:
        meta_loader = MetaLoader(
            dataloaders,
            accum_steps=args.gradient_accumulation_step,
            distributed=args.distributed,
            device=device,
            off_batch_task=args.off_batch_task
        )
        meta_loader = PrefetchLoader(meta_loader, device)

        if args.num_steps_per_epoch!=-1:
            meta_loader.num_batches = args.num_steps_per_epoch
    else:
        return dataloaders, agents
    return meta_loader, agents


def build_dataloader(dataset, distributed, training, batch_size, num_workers):
    if distributed:
        size = dist.get_world_size()
        sampler = DistributedSampler(
            dataset, num_replicas=size, rank=dist.get_rank(), shuffle=training
        )
        pre_epoch = sampler.set_epoch
    else:
        # not distributed
        if training:
            sampler: Union[
                RandomSampler, SequentialSampler, DistributedSampler
            ] = RandomSampler(dataset)
            # sampler = SequentialSampler(dataset)  # Debug Mode
        else:
            sampler = SequentialSampler(dataset)

        size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        pre_epoch = lambda e: None

        # DataParallel: scale the batch size by the number of GPUs
        if size > 1:
            batch_size *= size

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_batch,
    )
    loader.num_batches = len(loader)

    return loader, pre_epoch


class MetaLoader:
    """wraps multiple data loaders"""

    def __init__(
        self, loaders, accum_steps: int = 1, distributed: bool = False, device=None, off_batch_task: bool = False,
    ):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.name2pre_epoch = {}
        self.names: List[str] = []
        ratios: List[int] = []

        self.num_batches = 0
        self.off_batch_task = off_batch_task

        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r, p = l
            elif isinstance(l, DataLoader):
                r = 1
                p = lambda e: None
            else:
                raise ValueError()
            self.names.append(n)
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2pre_epoch[n] = p
            ratios.append(r)

            self.num_batches += l.num_batches

        self.accum_steps = accum_steps
        self.device = device
        self.sampling_ratios = torch.tensor(ratios).float().to(self.device)
        self.distributed = distributed
        self.step = 0
        self.epoch_id = 0  

    def get_dataset(self, name):
        return self.name2loader[name].dataset

    def __iter__(self) -> Iterator[Tuple]:
        """this iterator will run indefinitely"""
        task_id = None
        self.step = 0
        while True:
            # if self.step % self.accum_steps == 0:
            task_id = torch.multinomial(self.sampling_ratios, 1)
            if self.distributed and not self.off_batch_task:
                # make sure all process is training same task
                dist.broadcast(task_id, 0)
            self.step += 1
            task = self.names[task_id.cpu().item()]
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:

                self.epoch_id += 1
                # In distributed mode, calling the set_epoch() method at the beginning of each epoch
                # before creating the DataLoader iterator is necessary to make shuffling work properly
                # across multiple epochs. Otherwise, the same ordering will be always used.
                self.name2pre_epoch[task](self.epoch_id)
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch


def move_to_cuda(batch: Union[List, Tuple, Dict, torch.Tensor], device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, list):
        return [move_to_cuda(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_cuda(t, device) for t in batch)
    elif isinstance(batch, dict):
        return {n: move_to_cuda(t, device) for n, t in batch.items()}
    return batch


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    """
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.num_batches = self.loader.num_batches

    def get_dataset(self):
        return self.loader.dataset

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        self.batch = move_to_cuda(self.batch, self.device)

    def next(self, it):
        batch = self.batch
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method

