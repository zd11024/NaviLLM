import os
import json
import random
import torch
from tqdm import tqdm
from tools.common_utils import all_gather
from typing import Dict
from tools.trie import Trie
from transformers import LlamaTokenizer


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
        if name in ["ScanQA", "LLaVA"]:
            lm_loss = model("3dqa", batch, agent).loss
            lm_loss *= loss_coef / args.gradient_accumulation_step
            lm_loss.backward()

            loss_metric.accumulate(lm_loss.item() * args.gradient_accumulation_step)
            loss_stats[name].accumulate(lm_loss.item() * args.gradient_accumulation_step)
        else:
            if stage=='pretrain' or step%2==0:
                #################### imitation learning ####################
                ml_loss, _ = agent.rollout(
                    args, name, global_cfg.Optim, batch,
                    model=model, criterion=criterion, dataset=dataset,
                    feedback="teacher", train_ml=loss_coef * args.teacher_forcing_coef,
                    entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
                )

                loss_metric.accumulate(ml_loss.item() * args.gradient_accumulation_step)
                loss_stats[name].accumulate(ml_loss.item() * args.gradient_accumulation_step)
            else:
                #################### dagger training ####################
                sample_loss, _ = agent.rollout(
                    args, name, global_cfg.Optim, batch,
                    model=model, criterion=criterion, dataset=dataset,
                    feedback="sample", train_ml=loss_coef,
                    entropy_metric=entropy_metric, instr_pred_metric=instr_pred_metric
                )

                loss_metric.accumulate(sample_loss.item() * args.gradient_accumulation_step)
                loss_stats[name].accumulate(sample_loss.item() * args.gradient_accumulation_step)
        
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
        # We rely on env showing the entire batch before repeating anything
        looped = False
        pbar = tqdm(loader, disable=args.rank!=0)
        if name in ["ScanQA"]:
            preds = []
            for i, batch in enumerate(pbar):
                generation_kwargs = {
                    "do_sample": args.do_sample,
                    "temperature": args.temperature,
                    "max_new_tokens": 20
                }
                outputs = model("3dqa", batch, agent, training=False, **generation_kwargs)
                generated_sentences = outputs["generated_sentences"]
                for i in range(len(batch["question"])):
                    preds.append({
                        "scene_id": batch["scene_id"][i],
                        "question_id": batch["question_id"][i],
                        "generated_sentences": [generated_sentences[i].lower().strip()]
                    })
        else:
            results = {}
            trie = None
            if name in ['EQA']:
                if hasattr(model, 'module'):
                    tokenizer = model.module.lang_model.tokenizer
                else:
                    tokenizer = model.lang_model.tokenizer

                trie = Trie(tokenizer.bos_token_id, tokenizer.eos_token_id)
                for word in dataset.answer_vocab:
                    token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
                    if isinstance(tokenizer, LlamaTokenizer):
                        token_ids = [tokenizer.bos_token_id] + token_ids
                    trie.insert(token_ids)

            for i, batch in enumerate(pbar):
                ml_loss, traj = agent.rollout(
                    args, name, global_cfg.Optim, batch,
                    model=model, criterion=criterion, dataset=dataset,
                    feedback= "sample" if args.do_sample else "argmax", train_ml=None,
                    entropy_metric=entropy_metric, instr_pred_metric=None,
                    validate=True, trie=trie
                )

                for s_traj in traj:
                    if s_traj['instr_id'] in results:
                        looped = True
                    else:
                        ml_loss = 0
                        results[s_traj['instr_id']] = s_traj
     
                # Caldulate oracle prediction answer
                if name in ["EQA"]:
                    _, oracle_traj = agent.rollout(
                        args, name, global_cfg.Optim, batch,
                        model=model, criterion=criterion, dataset=dataset,
                        feedback="teacher", train_ml=1,
                        entropy_metric=entropy_metric, instr_pred_metric=None,
                        validate=True, trie=trie
                    )

                    for s_traj in oracle_traj:
                        results[s_traj['instr_id']]['oracle_pred_answer'] = s_traj['generated_sentences']

                if looped:
                    break

            # [MULTI-GPU] gather all prediction results from ALL GPU
            preds = get_results(results)  
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


def get_results(pred_results, detailed_output=False):
    pred_output = []
    for k, v in pred_results.items():
        ret = {
            'instr_id': k,
            'trajectory': v['path']
        }
        # scan_qa
        if 'answer' in v:
            ret.update({
                'pred_answer': v['generated_sentences'],
                'oracle_pred_answer': v.get('oracle_pred_answer', ''),
                'gt_answer': v['answer'],
            })
        
        # obj nav
        if 'pred_objid' in v:
            ret.update({
                'pred_objid': v['pred_objid'],
                'pred_obj_direction': v['pred_obj_direction']
            })
        pred_output.append(ret)

    return pred_output


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