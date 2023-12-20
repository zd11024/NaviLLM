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