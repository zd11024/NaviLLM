import numpy as np
import torch
from collections import defaultdict
from contextlib import nullcontext
from models.graph_utils import GraphMap
from typing import List

def rollout(
        args,
        name,
        config,
        batch_dict,
        model,
        criterion,
        dataset,
        feedback,
        train_ml,
        nav_agent,
        entropy_metric,
        instr_pred_metric,
        validate=False,
        **kwargs
):
    """
    :param args:
    :param name: task name
    :param config:
    :param batch_dict:
    :param model:
    :param criterion:
    :param dataset:
    :param feedback:
    :param train_ml:
    :param nav_agent:
    :param entropy_metric:
    :param validate:
    :return:
    """
    obs = batch_dict["observations"]
    envs = batch_dict["env"]
    data_type = batch_dict['data_type']

    max_action_len = config.val_max_action_len[name] if validate else config.train_max_action_len[name]

    nav_agent.update_scanvp_cands(obs)
    batch_size = len(obs)

    # build graph: keep the start viewpoint
    gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
    for i, ob in enumerate(obs):
        gmaps[i].update_graph(ob)

    traj = [{
        'instr_id': ob['instr_id'],
        'path': [[ob['viewpoint']]],
        'details': {},
    } for ob in obs]

    # Initialization the tracking state
    ended = np.array([False] * batch_size)
    just_ended = np.array([False] * batch_size)

    # instructions = language_variable(obs, data_type=batch_dict['data_type'])
    instructions = [ob["instruction"] for ob in obs]

    history = []
    hist_vis = []
    for idx in range(len(instructions)):
        history.append([])
        hist_vis.append([])

    entropys = []
    ml_loss, cnt_loss = 0., 0.
    flag = False

    for t in range(max_action_len):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # multi-gpu
            if ended.all() or t == max_action_len - 1:
                flag = True
                context = nullcontext
            else:
                context = model.no_sync
        else:
            # single-gpu
            if ended.all() or t == max_action_len - 1:
                flag = True
                context = nullcontext
            else:
                context = nullcontext

        with context():
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = nav_agent.panorama_feature_variable_object(obs)
            panorama_out = model('panorama', pano_inputs)
            pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)  # [B, D=768]

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    update_avg_pana_embeds = avg_pano_embeds[i].detach()  # update average features for gmap.
                    gmap.update_node_embed(i_vp, update_avg_pana_embeds, rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            update_pano_embeds = pano_embeds[i, j].detach()
                            gmap.update_node_embed(i_cand_vp, update_pano_embeds)

            # navigation policy
            nav_inputs = nav_agent.nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                nav_agent.nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )

            nav_inputs.update({
                'view_lens': pano_inputs['view_lens'],
                'instruction': instructions,
                'history': history,
                'hist_vis': hist_vis,
                'data_type': data_type
            })

            in_progress = torch.tensor(ended).logical_not()
            if ended.all():
                in_progress[0] = True

            reshaped_nav_inputs = {}
            for k, v in nav_inputs.items():
                if isinstance(v, torch.Tensor):
                    reshaped_nav_inputs[k] = v[in_progress]
                elif isinstance(v, List):
                    reshaped_nav_inputs[k] = [item for i, item in enumerate(v) if in_progress[i]]
                elif v is None:
                    reshaped_nav_inputs[k] = v
                else:
                    raise NotImplementedError

            reshaped_nav_outs = model('navigation', reshaped_nav_inputs)
            nav_outs = {}
            for k, v in reshaped_nav_outs.items():
                if isinstance(v, torch.Tensor):
                    nav_outs[k] = torch.zeros((batch_size,)+v.shape[1:], dtype=v.dtype, device=v.device)
                    nav_outs[k][in_progress] = v
                elif v is None:
                    nav_outs[k] = v
                else:
                    raise NotImplementedError(f'{type(v)} is not Implemented.')

            # dynamic fusion
            nav_logits = nav_outs['fuse_logits']
            nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits / args.temperature, 1)

            imitation_learning = feedback == 'teacher'
            if 'r2r' in data_type:
                nav_targets = nav_agent.teacher_action_r4r(
                    obs, nav_vpids, ended,
                    visited_masks=nav_inputs['gmap_visited_masks'],
                    imitation_learning=imitation_learning, t=t, traj=traj
                )
            else:
                nav_targets = nav_agent.teacher_action(
                    obs, nav_vpids, ended,
                    visited_masks=nav_inputs['gmap_visited_masks'],
                )
            # Imitation Learning
            if train_ml is not None:
                # [1] Supervised training

                ############# Single-Step Loss #############
                cnt_loss += criterion(nav_logits, nav_targets) * train_ml / batch_size / args.gradient_accumulation_step

                ml_loss += cnt_loss.detach()

                ########### Single-Step Backward ###########
                if not validate:
                    cnt_loss.backward()
                cnt_loss = 0.

            # Determinate the next navigation viewpoint
            if feedback == 'teacher':  # imitation learning
                a_t = nav_targets  # teacher forcing
            elif feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs.float())
                entropy_metric.accumulate(c.entropy().sum().item()/ batch_size)  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            elif feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            else:
                print(feedback)
                raise NotImplementedError

            for idx in range(len(a_t)):
                if a_t[idx] == -100:
                    continue
                history[idx] += ['<hist>']
                hist_vis[idx].append(nav_outs['fuse_embeds'][idx][a_t[idx]])

            if not validate:
                # if feedback == 'teacher' or feedback == 'sample':  # in training
                assert feedback in ['teacher', 'sample'], "Feedback must be either `teacher' or `sample' in training. "
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0
            
            ########### Object Prediction Sub-task ###########
            if (data_type[0] in ['soon', 'reverie']) and args.enable_og and flag:
                # graph representation
                pano_inputs = nav_agent.panorama_feature_variable_object(obs)
                panorama_out = model('panorama', pano_inputs)

                if 'obj_embeds' not in panorama_out:
                    pano_embeds = panorama_out['pano_embeds']
                    panorama_out.update({
                        "obj_embeds": torch.zeros((pano_embeds.shape[0], 0, pano_embeds.shape[2]), dtype=pano_embeds.dtype, device=pano_embeds.device),
                        "obj_masks": torch.zeros((pano_embeds.shape[0], 0), dtype=torch.int64, device=pano_embeds.device),
                        "obj_loc_fts": torch.zeros((pano_embeds.shape[0], 0, 7), dtype=pano_embeds.dtype, device=pano_embeds.device)
                    })

                nav_inputs.update({
                    'obj_embeds': panorama_out['obj_embeds'],
                    'obj_masks': panorama_out['obj_masks'],
                    'obj_loc_fts': panorama_out['obj_loc_fts']
                })

                nav_inputs.update({
                    'view_lens': pano_inputs['view_lens'],
                    'instruction': instructions,
                    'history': history,
                    'hist_vis': hist_vis,
                    'data_type': data_type
                })

                obj_logits = model('grounding', nav_inputs)['obj_logits']
                obj_targets = nav_agent.teacher_object(obs)

                if not validate:
                    obj_loss = criterion(obj_logits, obj_targets) * args.obj_loss_coef / batch_size / args.gradient_accumulation_step
                    obj_loss.backward()
                    ml_loss += obj_loss.detach()

                # update obj results
                for i, gmap in enumerate(gmaps):
                    i_vp = obs[i]['viewpoint']
                    i_objids = obs[i]['obj_ids']
                    i_obj_logits = obj_logits[i, 1:]
                    if 'obj_directions' in obs[i]:
                        traj[i].update({
                            'pred_objid': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                            'pred_obj_direction': obs[i]['obj_directions'][torch.argmax(i_obj_logits)] if len(
                                i_objids) > 0 else None,
                        })
                    else:
                        traj[i].update({
                            'pred_objid': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                            'pred_obj_direction': None,
                        })

            ########### Fine-grained R2R Sub-task ###########
            enable_fgr2r = (feedback == 'teacher') and (not flag) and (not a_t_stop[0]) and (data_type[0]=='r2r') and (not validate) and 'fg_instruction' in ob and args.enable_fgr2r
            if enable_fgr2r:
                pano_inputs = nav_agent.panorama_feature_variable_12views(obs)
                panorama_out = model('panorama', pano_inputs)
                pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                nav_inputs = nav_agent.nav_gmap_variable(obs, gmaps)
                nav_inputs.update(
                    nav_agent.nav_vp_variable(
                        obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                        pano_inputs['view_lens'], pano_inputs['nav_types'],
                    )
                )
                nav_inputs['instruction'] = ['where are we going with direction ({}) ?'.format(idx) for idx in nav_targets]
                nav_inputs["data_type"] = ['fgr2r' for idx in nav_targets]
                nav_inputs['answer'] = [ob['fg_instruction'][ob['fg_view'][t]] for ob in obs]
                nav_inputs['hist_vis'] = [[] for idx in nav_targets]
                nav_inputs['history'] = [[] for idx in nav_targets]

                output = model('sum', nav_inputs, training=not validate, **kwargs)
                if not validate:
                    lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                    lm_loss.backward()
                    instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                    ml_loss += lm_loss.detach()
         
            ########### Navigation Summarization Sub-task ###########
            if data_type[0] == 'eqa':
                enable_summarize = flag
            elif data_type[0] in ['r2r', 'soon', 'reverie', 'r2r_aug', 'reverie_aug']:
                enable_summarize = (feedback == 'teacher' or feedback == 'argmax') and flag and args.enable_summarize and (not validate or args.mode=='test')
            elif data_type[0] in ['cvdn']:
                enable_summarize = False
            else:
                raise NotImplementedError

            if enable_summarize:  # gen loss
                
                pano_inputs = nav_agent.panorama_feature_variable_12views(obs)
                panorama_out = model('panorama', pano_inputs)
                pano_embeds, pano_masks = panorama_out['pano_embeds'], panorama_out['pano_masks']
                nav_inputs = nav_agent.nav_gmap_variable(obs, gmaps)
                nav_inputs.update(
                    nav_agent.nav_vp_variable(
                        obs, gmaps, pano_embeds, pano_masks, pano_inputs['cand_vpids'],
                        pano_inputs['view_lens'], pano_inputs['nav_types'],
                    )
                )
                
                nav_inputs['instruction'] = [ob["instruction"] for ob in obs]
                nav_inputs['history'] = history
                nav_inputs['hist_vis'] = hist_vis
                nav_inputs["data_type"] = data_type
                nav_inputs['answer'] = [ob.get('answer', '') for ob in obs]

                output = model('sum', nav_inputs, training=not validate, **kwargs)
                if not validate:
                    lm_loss = output["loss"] * args.gen_loss_coef / batch_size / args.gradient_accumulation_step
                    lm_loss.backward()
                    instr_pred_metric.accumulate(lm_loss.detach().item() * args.gradient_accumulation_step)
                    ml_loss += lm_loss.detach()
                else:
                    for i in range(batch_size):
                        generated_sentences = output["generated_sentences"]
                        traj[i]['generated_sentences'] = generated_sentences[i]
                        traj[i]['answer'] = nav_inputs['answer'][i]

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                # TODO
                if False and data_type[i] == 'eqa':
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == max_action_len - 1):
                        cpu_a_t.append(None)
                        just_ended[i] = True
                    else:
                        cpu_a_t.append(nav_vpids[i][a_t[i]])

            # Make action and get the new state
            nav_agent.make_equiv_action(cpu_a_t, gmaps, obs, traj=traj, env=envs)

            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))

            # get new observation and update graph
            new_obs = []
            for b_i in range(batch_size):
                # TODO
                if False and data_type[b_i] == 'eqa':
                    raise NotImplementedError
                else:
                    new_obs.append(
                        dataset.get_obs(
                            items=[batch_dict['item'][b_i]],
                            env=envs[b_i], data_type=data_type[b_i]
                        )[0]
                    )
            obs = new_obs

            nav_agent.update_scanvp_cands(obs)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            if flag:
                break

    return ml_loss, traj
