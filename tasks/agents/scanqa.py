from .base_agent import BaseAgent


class ScanQAAgent(BaseAgent):
    name = "scanqa"

    def get_prompt(self, task, *args, **kwargs):
        if task == '3dqa':
            return self.get_3dqa_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_3dqa_prompt(self, ques, cand_num):
        obs_text = ' '.join(["({}) <cand>".format(i) for i in range(cand_num)])
        prompt = "Please answer questions based on the observation.\n" + \
            "The following is the Observation, which includes multiple images from different locations.\n" + \
            "### Observation: {} \n".format(obs_text) + \
            "### Question: {}\n".format(ques) + \
            "### Answer: "
        return prompt