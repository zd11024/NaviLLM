import os
import json
import torch
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import torch.nn as nn
from tools.common_utils import all_gather
from tools.parser import read_args, random_seed
from tasks.loaders import create_dataloaders
from tasks.feature_db import create_feature_db, create_object_feature_db
from models.nav_model import NavModel
from tools.optims import dist_models, save_checkpoint
from tools.trie import Trie

class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        if self.num == 0:
            return 0
        return self.total / self.num


def train_one_epoch(
        args,
        global_cfg,
        model,
        optimizer,
        lr_scheduler,
        criterion,
        dataloaders,
        agents,
        epoch,
        logger,
        stage='multi'
):

    model.train()
    entropy_metric = Metrics()
    loss_metric = Metrics()
    instr_pred_metric = Metrics()

    num_batches_per_epoch = dataloaders.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    pbar = tqdm(
        range(dataloaders.num_batches),
        disable=args.rank!=0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    
    dataset_cfg = global_cfg.Pretrain if stage=='pretrain' else global_cfg.Multi
    loss_stats = {k: Metrics() for k in dataset_cfg.SOURCE}

    for step, (name, batch) in enumerate(dataloaders):
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)
        # perform embodied tasks
        # the actual batch_size equals to args.batch_size * world_size * (args.gradient_accumulation_step)
        dataset = dataloaders.loader.get_dataset(name)
        agent = agents.get(name)
        loss = agent.train(
            name,
            batch,
            args,
            global_cfg,
            model=model,
            criterion=criterion,
            dataset=dataset,
            step=step,
            entropy_metric=entropy_metric,
            instr_pred_metric=instr_pred_metric
        )
        loss_metric.accumulate(loss.item())
        loss_stats[name].accumulate(loss.item())

        if (step+1) % args.gradient_accumulation_step==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.)
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()

        if args.rank == 0:
            verbose_dict = dict(
                step=step,
                name=name,
                # index=batch['sample_idx'],
                loss=loss_metric.average,
                entropy=entropy_metric.average,
                instr_pred_metric=instr_pred_metric.average,
                lr=lr_scheduler.get_last_lr()[0],
            )
            for k in dataset_cfg.SOURCE:
                verbose_dict[k] = loss_stats[k].average
            pbar.set_postfix(verbose_dict)
            pbar.update()

        if step == num_batches_per_epoch-1:
            logger.info("***** train [{}] epoch *****wo".format(epoch))
            train_stat_str = 'Loss: %.2f\n' % loss_metric.average
            train_stat_str += "Instr_pred: %.2f\n" % instr_pred_metric.average
            for task in dataset_cfg.SOURCE:
                train_stat_str += "%s: %.2f\n" % (task, loss_stats[task].average)
            logger.info(train_stat_str)
            break


@torch.no_grad()
def val_one_epoch(
        args,
        global_cfg,
        model,
        optimizer,
        criterion,
        dataloaders,
        agents,
        epoch,
        logger,
) -> Dict[str, Dict[str, float]]:

    model.eval()
    entropy_metric = Metrics()

    loss_str = "\n[Eval] {} epoch {}\n".format(args.validation_split, epoch)
    task_results = {}
    for name, loader in dataloaders.items():
        logger.info("***** validate {} split on {} task *****".format(args.validation_split, name))
        dataset = dataloaders[name].get_dataset()
        agent = agents[name]
        preds = agent.validate(
            name,
            args,
            global_cfg,
            model,
            loader,
            entropy_metric=entropy_metric
        )

        all_preds = all_gather(preds)
        all_preds = merge_dist_results(all_preds)

        if args.rank == 0 and not args.validation_split.startswith('test'):
            score_summary, item_metrics = dataset.eval_metrics(all_preds, logger=logger, name=name)

            task_results[name] = score_summary
            loss_str += "\n [Eval] dataset=[{}] \n".format(name)
            for metric, val in score_summary.items():
                if metric == 'sr':
                    loss_str += '\n[Eval] ||| %s: %.2f' % (metric, val)
                else:
                    loss_str += ', %s: %.2f' % (metric, val)
        
        if args.rank== 0 and args.save_pred_results:
            dataset.save_json(
                all_preds, 
                os.path.join(args.output_dir, f"{name}_{args.validation_split}.json"),
                item_metrics=item_metrics if args.save_detail_results else None
            )


    logger.info(loss_str)
    
    return task_results



def merge_dist_results(results):
    outs = []
    for res in results:
        outs.extend(res)
    return outs


def calc_overall_score(results, cfg):
    score = 0.
    for task in results:
        if task not in cfg.Multi.SOURCE:
            continue
        if task == 'R2R':
            score += results[task]['spl'] / 60
        elif task == 'REVERIE':
            score += results[task]['spl'] / 36.63
        elif task == 'CVDN':
            pass
        elif task == 'SOON':
            score += results[task]['spl'] / 26.58
        elif task == 'EQA':
            pass
        elif task == "ScanQA":
            pass
        else:
            raise NotImplementedError(f"The method for calculating the score of {task} is not Implemented.")

    return score


def main():
    args, global_cfg, logger, device_id = read_args()
    random_seed(args.seed + args.rank)

    ##################### DATASET #####################
    feat_db = create_feature_db(global_cfg.Feature.feature_database, global_cfg.Feature.image_feat_size, args)
    obj_feat_db = create_object_feature_db(global_cfg.Feature.object_database, global_cfg.Feature.obj_feat_size, args)
    # Initialize train dataloader
    if args.mode == "train":
        train_dataloaders, train_agents = create_dataloaders(
            args, global_cfg, logger,
            training=True, device=device_id, feat_db=feat_db, obj_feat_db=obj_feat_db, stage=args.stage
        )
    # Initialize val dataloader
    val_dataloaders, val_agents = create_dataloaders(
        args, global_cfg, logger,
        training=False, device=device_id, feat_db=feat_db, obj_feat_db=obj_feat_db, stage="multi"
    )

    # Model
    model = NavModel(args, logger, global_cfg.Model)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='sum')

    model, optimizer, resume_from_epoch, lr_scheduler = dist_models(args, model, logger)
    if args.mode=="test":
        logger.info("**************************** Test ****************************")
        results = val_one_epoch(
            args, global_cfg, model, optimizer, criterion, val_dataloaders, val_agents, resume_from_epoch, logger
        )
    elif args.mode == "train":
        logger.info("**************************** Train ****************************")

        best_results, best_score = None, None
        history_scores = []
        for epoch in range(resume_from_epoch, args.num_epochs):
            # training
            train_one_epoch(
                args, global_cfg, model, optimizer, lr_scheduler, criterion, train_dataloaders, train_agents, epoch, logger, stage=args.stage
            )

            # evaluation
            results = val_one_epoch(
                args, global_cfg, model, optimizer, criterion, val_dataloaders, val_agents, epoch, logger
            )

            if args.rank==0:
                score = calc_overall_score(results, global_cfg)
                history_scores.append(score)
                should_save_checkpoint = False

                if best_results is None or score > best_score:
                    best_results = results
                    best_score = score
                    should_save_checkpoint = args.max_saved_checkpoints > 0
                
                logger.info(f"Current Score: {score}")
                logger.info(f"Best Score: {best_score}")

                if args.stage=='multi':
                    # Save the best
                    if should_save_checkpoint:
                        if len(history_scores) > args.max_saved_checkpoints:
                            sorted_scores = sorted(enumerate(history_scores), key=lambda x: x[1], reverse=True)
                            
                            remove_epoch = sorted_scores[args.max_saved_checkpoints][0]
                            remove_model_path = Path(args.output_dir) / f"epoch_{remove_epoch}.pt"
                            if os.path.exists(remove_model_path):
                                os.remove(remove_model_path)
                                logger.info(f"Remove Checkpoint at Epoch {remove_epoch}...")

                        model_path = Path(args.output_dir) / f"epoch_{epoch}.pt"
                        save_checkpoint(model, model_path)
              
                elif args.stage=='pretrain' and (epoch+1)%args.save_ckpt_per_epochs==0:
                    model_path = Path(args.output_dir) / f"pretrain_{epoch}.pt"
                    save_checkpoint(model, model_path)

              
            if args.save_latest_states:
                # Save the latest if args.save_latest_states is True
                model_path = Path(args.output_dir) / f"latest.pt"
                save_checkpoint(model, model_path, optimizer, epoch, save_states=True)
        
        # print best results
        if args.rank == 0:
            logger.info(f"Best Results:")
            logger.info(best_results)

if __name__ == '__main__':
    main()
