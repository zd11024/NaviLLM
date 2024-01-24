from .mp3d_agent import MP3DAgent

class R2RAgent(MP3DAgent):
    name = "r2r"

    def get_prompt(self, task, *args, **kwargs):
        if task == 'navigation':
            return self.get_navigation_prompt(*args, **kwargs)
        elif task == 'summarization':
            return self.get_summarization_prompt(*args, **kwargs)
        elif task == 'embodied_qa':
            return self.get_embodied_qa_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_navigation_prompt(self, instruction, hist_num, cand_num, cls_token):
        # Task
        if self.remove_task_hint:
            prompt = "### Instruction: {} \n".format(instruction)
        else:
            prompt = '### Instruction: Navigate following the instruction. {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        if self.remove_output_hint:
            prompt += ""
        else:
            prompt += 'Compare the History and Instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location.\n'
        prompt += '### Output: {}'.format(cls_token)

        return prompt

    def get_summarization_prompt(self, instruction, hist_num, cand_num):
        # Task
        if self.remove_task_hint:
            prompt = "### Instruction: Generate the task. \n"
        else:
            prompt = f'### Instruction: Predict the fine-grained instruction based on your previous history and current location. Fine-grained instructions contain commands for each individual step. \n'
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        if cand_num != 0:
            prompt += 'Following is the Observation, which contains panoramic views at your current location.\n'
            obs_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
            prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        if self.remove_output_hint:
            prompt += ''
        else:
            prompt += 'Please generate the step-by-step instruction.\n'
        prompt += '### Answer: '

        return prompt

    def get_embodied_qa_prompt(self, instruction, hist_num, cand_num):  
        # Task
        prompt = f'### Instruction: answer the question. \n'
        # History
        if hist_num !=0:
            prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
            hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
            prompt += '### History: {}\n'.format(hist_text)
        # Observation
        if cand_num != 0:
            prompt += 'Following is the Observation, which contains panoramic views at your current location.\n'
            obs_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
            prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += '### Question: {}\n'.format(instruction)
        prompt += '### Answer: '
        
        return prompt


class R2RAugAgent(R2RAgent):
    name = "r2r_aug"