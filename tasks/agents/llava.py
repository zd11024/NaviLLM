from tqdm import tqdm
from .base_agent import BaseAgent

class LLaVAAgent(BaseAgent):
    name = "llava"

    def get_prompt(self, task, *args, **kwargs):
        if task == "3dqa":
            return self.get_3dqa_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_3dqa_prompt(self, ques, cand_num):
        prompt = "### Image: <cand>\n" + \
            "### Instruction: {}\n".format(ques) + \
            "### Output: "
        return prompt
    
    def train(
        self,
        name,
        batch,
        args,
        config,
        model,
        **kwargs
    ):
        assert name in ["ScanQA", "LLaVA"], 'The task name must be in [ScanQA, LLaVA]'
        dataset_cfg = config.Pretrain if args.stage=='pretrain' else config.Multi
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)
        # construct prompt
        prompts = []
        batch_size = len(batch["question"])
        # update prompts
        batch["prompts"] = self.prepare_prompts(batch)

        # forward the model
        lm_loss = model("3dqa", batch).loss
        lm_loss *= loss_coef / args.gradient_accumulation_step
        lm_loss.backward()

        return lm_loss * args.gradient_accumulation_step

    
    def validate(
        self,
        name,
        args,
        config,
        model,
        loader,
        **kwargs,
    ):
        assert name in ["ScanQA"]
        preds = []
        pbar = tqdm(loader, disable=args.rank!=0)
        for i, batch in enumerate(pbar):
            generation_kwargs = {
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "max_new_tokens": 20
            }
            batch["prompts"] = self.prepare_prompts(batch)
            outputs = model("3dqa", batch, training=False, **generation_kwargs)
            generated_sentences = outputs["generated_sentences"]
            for i in range(len(batch["question"])):
                preds.append({
                    "scene_id": batch["scene_id"][i],
                    "question_id": batch["question_id"][i],
                    "generated_sentences": [generated_sentences[i].lower().strip()]
                })
    
        return preds
    
    def prepare_prompts(self, batch):
        prompts = []
        for bn in range(len(batch["question"])):
            prompts.append(
                self.get_prompt(
                    '3dqa',
                    ques = batch["question"][bn],
                    cand_num = batch["features"][bn].shape[0]
                )
            )
        return prompts