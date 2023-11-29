import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from transformers import OPTForCausalLM, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.logits_process import LogitsProcessor
from tools.trie import Trie


class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, trie: Trie):
        self.node_states = None
        self.trie = trie
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        if self.node_states is None:
            self.node_states = [self.trie.root for bn in range(batch_size)]
        else:
            for bn in range(batch_size):
                w = input_ids[bn, -1].item()
                self.node_states[bn] = self.trie.get_next_node(self.node_states[bn], w)
        
        masks = torch.zeros_like(scores, dtype=torch.bool).to(scores.device)
        for bn in range(batch_size):
            next_layer = self.trie.get_child_index(self.node_states[bn])
            masks[bn][next_layer] = True
        
        scores = scores.masked_fill(~masks, float('-inf'))
        return scores


class ModifiedLM:
    """
    This is base class for all ModifiedLM*
    """

    def __init__(self, extra_config):

        if extra_config.precision == 'fp16':
            self.model_type = torch.float16
        elif 'bf16' in extra_config.precision or 'bfloat16' in extra_config.precision:
            self.model_type = torch.bfloat16
        else:
            self.model_type = torch.float32

        self.model = self.model.to(self.model_type)
        self.lm_head = self.lm_head.to(self.model_type)

        # print("************ Use dtype: {} ************\n".format(self.model_type))

        # llama-7b dim=4096, bloom dim=1024,
        self.hidden_size = self.config.hidden_size


    def init_tokenizer(self, pretrained_model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left", truncation_side='left') if not isinstance(self.config, LlamaConfig) else LlamaTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left", truncation_side='left')

        self.cand_token = ['<cand>']
        self.hist_token = ['<hist>']
        self.obj_token = ['<obj>']
        self.cls_token = ['<cls_1>', '<cls_2>']
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.cand_token + self.hist_token + self.obj_token + self.cls_token}
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        self.cand_token_id = self.tokenizer.encode("".join(self.cand_token), add_special_tokens=False)
        self.hist_token_id = self.tokenizer.encode("".join(self.hist_token), add_special_tokens=False)
        self.obj_token_id = self.tokenizer.encode("".join(self.obj_token), add_special_tokens=False)
        self.cls_token_id = self.tokenizer.encode("".join(self.cls_token), add_special_tokens=False)
        self.special_token_ids = self.cand_token_id + self.hist_token_id + self.obj_token_id + self.cls_token_id
        
        self.resize_token_embeddings(len(self.tokenizer))

    def tokenize(self, text: str, add_special_tokens: bool=True):
        batch_text = self.tokenizer(
            text,
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            return_token_type_ids=True
        )
        return batch_text

    def forward(
        self, 
        input_ids,
        attention_mask, 
        labels=None,
        cand_vis=None, 
        hist_vis=None, 
        obj_vis=None, 
        **kwargs
    ):

        hist_locations = (input_ids >= self.hist_token_id[0]) & (input_ids <= self.hist_token_id[-1])
        cand_locations = (input_ids >= self.cand_token_id[0]) & (input_ids <= self.cand_token_id[-1])
        obj_locations = (input_ids >= self.obj_token_id[0]) & (input_ids <= self.obj_token_id[-1])

        inputs_embeds = self.get_input_embeddings()(input_ids)
        if cand_locations.sum() != 0:
            inputs_embeds[cand_locations] += cand_vis
        if hist_locations.sum() != 0:
            inputs_embeds[hist_locations] += hist_vis
        if obj_locations.sum() != 0:
            inputs_embeds[obj_locations] += obj_vis

        outputs = self.get_encoder()(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        # outputs = self.model.transformer(*input, **kwargs)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        logits_mask = torch.ones_like(logits, dtype=torch.bool).to(logits.device)
        logits_mask[:, :, self.special_token_ids] = False
        logits = logits.masked_fill(~logits_mask, float('-inf'))

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # logits = logits[cand_locations]
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,    # only store the last hidden states
            attentions=outputs.attentions,
        )
    

class ModifiedOPTForCasualLM(ModifiedLM, OPTForCausalLM):
    def __init__(self, config, extra_config):
        OPTForCausalLM.__init__(self, config)
        ModifiedLM.__init__(self, extra_config)
    
    def get_encoder(self):
        return self.model.decoder
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cand_vis=None, hist_vis=None, obj_vis=None, **kwargs
    ):
        model_inputs = OPTForCausalLM.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            **kwargs
        )
        if not past_key_values:
            for k in ['cand_vis', 'hist_vis', 'obj_vis']:
                model_inputs[k] = eval(k)

        return model_inputs



class ModifiedLlamaForCausalLM(ModifiedLM, LlamaForCausalLM):
    def __init__(self, config, extra_config):
        LlamaForCausalLM.__init__(self, config)
        ModifiedLM.__init__(self, extra_config)
    
    def get_encoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cand_vis=None, hist_vis=None, obj_vis=None, **kwargs
    ):
        model_inputs = LlamaForCausalLM.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            **kwargs
        )
        if not past_key_values:
            for k in ['cand_vis', 'hist_vis', 'obj_vis']:
                model_inputs[k] = eval(k)

        return model_inputs