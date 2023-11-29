
class NavPrompt:
    def get_prefix(self, instruction: str, data_type: str) -> str:
        if data_type in ["r2r", "r2r_aug"]:
            # walk towards the faucet sculpture, continue past it on left side and walk to the right of the marble cylinder to the doorway. Walk through the doorway to the right of the long rug, around the foot of the bed, and through the doorway on the left. Stop in front of the towel rack.
            return  'Navigate following the instruction. ' \
                + instruction
        elif data_type == 'soon':
            # 'I want to find a ceramic, rectangular and white sink, which is set in the washroom. The sink is under the mirror and next to the toilet. The washroom is inside the office, which is on the first floor and next to the living room.'
            return 'Find the described target. Target: ' \
                + instruction
        elif data_type in ["reverie", "reverie_aug"]:
            # Proceed to the office and turn on the ceiling fan
            return 'Go to the location to complete the given task. Task: ' \
                + instruction
        elif data_type == 'eqa':
            # what color is the counter in the dining room?
            # return 'Navigate to the object in the question. Question: ' \
            return 'Navigate following the instruction. Move to the object in "{}", and stop there.'.format(instruction.replace('?', ''))
        elif data_type == 'cvdn':
            # The goal room contains a fireplace.\nQuestion: where should i go?\nAnswer: If you turn around and keep heading then there is an open door on your right.\nQuestion: okay where to?\nAnswer: Go through the open doors on the right.
            return 'Find the described room according the given dialog. Target: ' \
                + instruction


    def get_suffix(self, data_type: str) -> str:
        suffix = ""
        if data_type in ["r2r", "r2r_aug"]:
            suffix = "Compare the History and Instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location.\n"
        elif data_type == "cvdn":
            suffix = "Understand the dialog in the Instruction and infer the current progress based on the History and dialog. Then select the correct direction from the candidates to go to the target location.\n"
        elif data_type in ["reverie", "reverie_aug"]:
            suffix = "Explore the scene to find out the targeted room and object. Then select the correct direction from the candidates to go to the target location.\n"
        elif data_type == "soon":
            suffix = "Nearby areas and objects can assist you in locating the desired room and object. Select the correct direction from the candidates to go to the target location.\n"
        elif data_type == "eqa":
            suffix = "Compare the History and Instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location.\n"
        else:
            raise NotImplementedError

        return suffix


    def get_prompt(
        self,
        instruction: str,
        hist_num: int,
        cand_num: int,
        data_type: str,
        cls_token: str,
    ) -> str:
        prompt = ""

        # prefix
        prefix = self.get_prefix(instruction, data_type)
        prompt += '### Instruction: {} \n'.format(prefix)

        # history
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)

        # candidate
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        cand_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(cand_text)

        # suffix
        prompt += self.get_suffix(data_type)
        prompt += '### Output: {}'.format(cls_token)
        
        return prompt
    

    def get_grounding_prompt(
        self,
        instruction: int,
        hist_num: int,
        cand_num: int,
        data_type: str,
        cls_token: str,
    ) -> str:
        prompt = "Select the target object from the candidate objects based on the instruction and history.\n"

        # prefix
        prefix = self.get_prefix(instruction, data_type)
        prompt += '### Instruction: {} \n'.format(prefix)

        # history
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)

        # candidate
        prompt += 'Following is the Object, which contains several objects that you could see at the current viewpoint, option (0) indicates not exist.\n'
        cand_text = ' '.join(['({}) <cand>'.format(i) if i>0 else '(0) not exist' for i in range(cand_num)])
        prompt += '### Object: {}\n'.format(cand_text)

        # suffix
        prompt += "Select the target object from the candidate objects according to the instruction.\n"
        prompt += '### Output: {}'.format(cls_token)

        return prompt



class SumPrompt:
    def get_prefix(self, instruction: str, data_type: str) -> str:
        prefix = ''
        if data_type in ["r2r", "r2r_aug"]:
            # walk towards the faucet sculpture, continue past it on left side and walk to the right of the marble cylinder to the doorway. Walk through the doorway to the right of the long rug, around the foot of the bed, and through the doorway on the left. Stop in front of the towel rack.
            prefix += 'Predict the fine-grained instruction based on your previous history and current location. Fine-grained instructions contain commands for each individual step.'
        elif data_type == 'soon':
            # 'I want to find a ceramic, rectangular and white sink, which is set in the washroom. The sink is under the mirror and next to the toilet. The washroom is inside the office, which is on the first floor and next to the living room.'
            prefix += 'Generate the target you want to find based on your previous history and current location. Describe both the target and its surroundings.'
        elif data_type in ["reverie", "reverie_aug"]:
            # Proceed to the office and turn on the ceiling fan
            prefix += 'Generate the task you need to complete based on your previous history and current location.'
        elif data_type == 'cvdn':
            # The goal room contains a fireplace.\nQuestion: where should i go?\nAnswer: If you turn around and keep heading then there is an open door on your right.\nQuestion: okay where to?\nAnswer: Go through the open doors on the right.
            prefix += 'Predict the conversation between you and a human based on your previous history and current location. In this dialogue, you ask the human for directions, and the human tells you how to navigate.'
        elif data_type == 'eqa':
            prefix += 'Answer the question according to the scene.'
        elif data_type == 'fgr2r':
            prefix += 'answer the question.'
        else:
            raise NotImplementedError
        return prefix

    def get_suffix(self, instruction: str, data_type: str) -> str:
        suffix = ""
        if data_type in ["r2r", "r2r_aug"]:
            suffix += 'Please generate the step-by-step instruction.\n'
        elif data_type == 'soon':
            suffix += 'Please predict both the target you want to find and its surroundings.\n'
        elif data_type in ["reverie", "reverie_aug"]:
            suffix += 'Please predict the task you need to complete.\n'
        elif data_type == 'cvdn':
            suffix += 'Please generate the dialogue between you and the human.\n'
        elif data_type == 'eqa':
            suffix += '### Question: {}\n'.format(instruction)
        elif data_type == 'fgr2r':
            suffix += '### Question: {}\n'.format(instruction)
        else:
            raise NotImplementedError
        return suffix


    def get_prompt(
        self,
        instruction: str,
        hist_num: int,
        cand_num: int,
        data_type: str,
    ) -> str:
        prompt = ''

        # prefix
        prefix = self.get_prefix(instruction, data_type)
        prompt += '### Instruction: {} \n'.format(prefix)

        # history
        if hist_num != 0:
            prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
            hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
            prompt += '### History: {}\n'.format(hist_text)

        # candidate
        if cand_num != 0:
            prompt += 'Following is the Observation, which contains panoramic views at your current location.\n'
            cand_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
            prompt += '### Observation: {}\n'.format(cand_text)

        # suffix
        prompt += self.get_suffix(instruction, data_type)
        prompt += '### Answer: '
        
        return prompt

class CaptionPrompt:
    def get_prompt(
        self,
        instruction: str,
        hist_num: int,
        cand_num: int,
        data_type: str,
    ) -> str:
        prompt = ''
        prompt += 'Describe the visual content of the scene in great detail.\n'
        
        if cand_num != 0:
            prompt += 'Following is the Observation, which contains panoramic views at your current location.\n'
            cand_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
            prompt += '### Observation: {}\n'.format(cand_text)
        
        prompt += 'Generate the detailed description for the scene.\n'
        prompt += '### Output: '
        return prompt