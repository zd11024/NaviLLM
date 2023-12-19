import os
from pathlib import Path
import torch.nn as nn
from tools.parser import read_args, random_seed
from tasks.loaders import create_dataloaders
from tasks.feature_db import create_feature_db, create_object_feature_db
from models.nav_model import NavModel
from tools.optims import dist_models, save_checkpoint
from core.pipeline import train_one_epoch, val_one_epoch, calc_overall_score


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
