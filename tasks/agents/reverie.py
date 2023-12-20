from .mp3d_agent import MP3DAgent

class REVERIEAgent(MP3DAgent):
    name = "reverie"

    def get_prompt(self, task, *args, **kwargs):
        if task == 'navigation':
            return self.get_navigation_prompt(*args, **kwargs)
        elif task == 'summarization':
            return self.get_summarization_prompt(*args, **kwargs)
        elif task == 'object_grounding':
            return self.get_object_grounding_prompt(*args, **kwargs)
        else:
            raise NotImplementedError
    
    def get_navigation_prompt(self, instruction, hist_num, cand_num, cls_token):
        # Task
        prompt = f'### Instruction: Go to the location to complete the given task. Task: {instruction} \n'
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += 'Explore the scene to find out the targeted room and object. Then select the correct direction from the candidates to go to the target location.\n'
        prompt += '### Output: {}'.format(cls_token)

        return prompt

    def get_summarization_prompt(self, instruction, hist_num, cand_num):
        # Task
        prompt = '### Instruction: Generate the task you need to complete based on your previous history and current location. \n'
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
        prompt += 'Please predict the task you need to complete.\n'
        prompt += '### Answer: '
        return prompt
        
    def get_object_grounding_prompt(self, instruction, hist_num, cand_num, cls_token):

        # Task
        prompt = "Select the target object from the candidate objects based on the instruction and history.\n"
        prompt += f'### Instruction: Go to the location to complete the given task. Task: {instruction} \n'

        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)

        # Observation
        prompt += 'Following is the Object, which contains several objects that you could see at the current viewpoint, option (0) indicates not exist.\n'
        cand_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) not exist' for i in range(cand_num)])
        prompt += '### Object: {}\n'.format(cand_text)

        # Output Hint
        prompt += "Select the target object from the candidate objects according to the instruction.\n"
        prompt += '### Output: {}'.format(cls_token)

        return prompt

class REVERIEAugAgent(REVERIEAgent):
    name = "reverie_aug"