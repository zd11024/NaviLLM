from .mp3d_agent import MP3DAgent

class EQAAgent(MP3DAgent):
    name = "eqa"

    def get_prompt(self, *args, **kwargs):
        if task == 'navigation':
            return self.get_navigation_prompt(*args, **kwargs)
        elif task == 'embodied_qa':
            return self.get_embodied_qa_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_navigation_prompt(self, instruction, hist_num, cand_num, cls_token):

        # Task
        prompt = '### Instruction: Navigate following the instruction. Move to the object in "{}", and stop there. \n'.format(instruction.replace('?', ''))
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += 'Compare the History and Instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location.\n'
        prompt += '### Output: {}'.format(cls_token)

        return prompt

    def get_embodied_qa_prompt(self, instruction, hist_num, cand_num):  
        # Task
        prompt = f'### Instruction: Answer the question according to the scene. \n'
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
        prompt += '### Question: {}\n'.format(instruction)
        prompt += '### Answer: '

        return prompt
