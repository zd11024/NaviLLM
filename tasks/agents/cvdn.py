from .mp3d_agent import MP3DAgent

class CVDNAgent(MP3DAgent):
    name = "cvdn"

    def get_prompt(self, task, *args, **kwargs):
        if task == 'navigation':
            return self.get_navigation_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_navigation_prompt(self, instruction, hist_num, cand_num, cls_token):
        # Task
        prompt = f'### Instruction: Find the described room according the given dialog. Target: {instruction} \n'
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += 'Understand the dialog in the Instruction and infer the current progress based on the History and dialog. Then select the correct direction from the candidates to go to the target location.\n'
        prompt += '### Output: {}'.format(cls_token)
        
        return prompt
