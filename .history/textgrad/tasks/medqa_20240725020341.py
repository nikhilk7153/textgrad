import platformdirs
from datasets import load_dataset
import json

from textgrad.variable import Variable
from textgrad.loss import MultiChoiceTestTime, MultiFieldTokenParsedEvaluation
from .base import Dataset
import re

def eval_string_based(response_text, correct_answer):
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
    
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == correct_answer else 0.0
    return score


def make_dataset(file_path):

    dataset = []

    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}

    with open(file_path) as file:
        for line in file:
            data = json.loads(line)
            dataset.append({"question": data["question"], "choices": [data["options"]["A"], data["options"]["B"], data["options"]["C"], data["options"]["D"]], "answer": data["answer_idx"]})

    return dataset 


# Below template is from https://github.com/openai/simple-evals/blob/main/common.py#L12
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

class MedQA(Dataset):
    def __init__(self, subset:str, root: str=None, split: str="test", *args, **kwargs):
        """
        MMLU dataset from HF."""
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
            
        self.root = root
        self.subset = subset
        assert split in ["train", "validation", "test"]
      
        
        self.data = []

        if split == "train":
            self.data = make_dataset("/Users/khandekarns/Downloads/data_clean/questions/US/4_options/phrases_no_exclude_train.jsonl")
        
        if split == "validation":
            self.data = make_dataset("/Users/khandekarns/Downloads/data_clean/questions/US/4_options/phrases_no_exclude_dev.jsonl")

        if split == "test":
            self.data = make_dataset("/Users/khandekarns/Downloads/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl")

        self.split = split
        self._task_description = 'You will answer multiple-choice questions. Think step by step.'
            
    def __getitem__(self, index):
        row = self.data[index]
        question = row["sent1"]
        #choices = row["choices"]
        # Choices will be a. Choice 1 b. Choice 2 ... etc
        #choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

        choices_str = "A. " + row["ending0"]
        choices_str += "\nB. " + row["ending1"]
        choices_str += "\nC. " + row["ending2"]
        choices_str += "\nD. " + row["ending3"]
  
        answer = chr(65+row["label"])
        question_prompt = f"Question: {question}\n\nChoices:\n\n{choices_str}"
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_default_task_instruction(self):
        return "Given a multiple choice question, the goal is to select the correct final answer from the choices."


class MedQAInstanceDataset(MedQA):
    def __init__(self, evaluation_api, subset:str, root: str=None, split: str="test", max_samples=-1):
        super().__init__(subset, root, split, max_samples)
        self.evaluation_api = evaluation_api

        
    def _get_instance_test_time_objective(self, question: str):
        evaluation_instruction = "Below is a multi-choice question and an answer. You are an expert scientist. Your job is to investigate the answer. Critically go through reasoning steps, consider your knowledge, and see if the answer is correct or if there are any critical mistakes."
        eval_fn = MultiChoiceTestTime(evaluation_instruction, engine=self.evaluation_api)
        def test_time_objective(instance: Variable):
            return eval_fn(question, instance)
        return test_time_objective
        

    def _legacy_get_instance_eval_fn(self, question_prompt: str, answer: str):
        role_descriptions = [
            "Question for the task",
            "Correct answer",
            "Prediction from the language model"
        ]
        eval_system_prompt = Variable("You are a language model that evaluates the accuracy of a prediction for a medical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")

        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and a prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=self.evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"],
            system_prompt=eval_system_prompt
        )
        
        answer_var = Variable(answer, requires_grad=False, role_description="Correct answer")
        question_var = Variable(question_prompt, requires_grad=False, role_description="Question for the task")
        
        def instance_eval_fn(instance):
            eval_output = eval_fn([question_var, answer_var, instance])
            return eval_fn.parse_output(eval_output)
        return instance_eval_fn

    def _get_instance_eval_fn(self, question_prompt: str, answer: str):
        eval_string_based_fn = lambda response: eval_string_based(response.value, answer)
        return eval_string_based_fn
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row["question"]
        choices_dict = dict(
                A=row["A"], B=row["B"], C=row["C"], D=row["ending3"], Question=question
            )
        question_prompt = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        # Choices will be a. Choice 1 b. Choice 2 ... etc
        answer = chr(65+ row["label"])
        # TODO: Make the two-way comparison class abstract enough.
        # TODO: How do we determine the role of the instances? We should be more consistent
        return question_prompt, answer, self._get_instance_test_time_objective(question_prompt), self._get_instance_eval_fn(question_prompt, answer)

    def get_default_task_instruction(self):
        return "Given a multiple choice question, the goal is to select the correct final answer from the choices."


