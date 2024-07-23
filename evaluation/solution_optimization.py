import re
import json
import argparse
import concurrent
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)
from statistics import multimode

import os

import textgrad as tg
from textgrad.tasks import load_instance_task


os.environ["AZURE_OPENAI_API_KEY"] = "a494edc84d714b6c8a12e7212974b793"
os.environ["AZURE_OPENAI_API_BASE"] = "https://bionlp-gpt4-wang.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "a494edc84d714b6c8a12e7212974b793"
os.environ["AZURE_OPENAI_ENDPOINT"] = "2024-03-01-preview"     


def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="MMLU_machine_learning", help="The task to evaluate the model on.")
    parser.add_argument("--engine", type=str, default="gpt-4o", help="The API to use for evaluation.")
    parser.add_argument("--max_iterations", type=int, default=3, help="The maximum number of iterations of test-time updates.")
    parser.add_argument("--num_threads", type=int, default=16, help="The number of threads to use for evaluation.")
    return parser.parse_args()


class MajorityVoting:
    def __init__(self):
        pass

    def __call__(self, predictions):
        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
        pred_labels = []
        for pred in predictions:
            match = re.search(ANSWER_PATTERN_MULTICHOICE, pred.value)
            extracted_answer = match.group(1) if match else None
            pred_labels.append(extracted_answer)
        
        modes = multimode(pred_labels)
        return tg.Variable(f"Answer: {modes[0]}", role_description="Majority ensemble")


def get_zeroshot_answer(question):
    """Getting the zero-shot answer from an LLM without optimizing the response at test time."""
    # The system prompt is from: https://github.com/openai/simple-evals/blob/main/sampler/chat_completion_sampler.py
    STARTING_SYSTEM_PROMPT = (
        "You are a medical expert. You are given a question with 4 option choices (A, B, C, or D). Please provide a step-by-step explanation explaining why three of the options are incorrect and one of them is correct. The last line of your output should be your final answer choice as 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.' "
    )

    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=False, role_description="system prompt to the language model")
    model = tg.BlackboxLLM(llm_engine, system_prompt)

    response = model(tg.Variable(question, requires_grad=False, role_description="question to the language model"))

    return response

def run_test_time_training(sample):
    performance_history = []
    question, answer, test_time_objective, instance_eval_fn = sample
    zero_shot_response = get_zeroshot_answer(question)
    
    instance_var = tg.Variable(zero_shot_response.value,
                               requires_grad=True,
                               role_description="precise explanation explaining why the 3 of the answer choices are incorrect and of one of them is correct? Prediction for the multiple choice question")
    
    # Evaluate the zero-shot response
    performance_history.append(int(instance_eval_fn(instance_var)))
    
    optimizer = tg.TextualGradientDescent(engine=llm_engine, 
                                          parameters=[instance_var], 
                                          constraints=["The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD."])

    predictions = []
    predictions.append(tg.Variable(
        instance_var.value,
        role_description=instance_var.role_description
    ))

    # Start test time training
    for _ in range(3):
        optimizer.zero_grad()
        # Compute the test time loss
        test_time_loss = test_time_objective(instance_var)
        test_time_loss.backward()
        optimizer.step()
        performance_history.append(instance_eval_fn(instance_var))
        predictions.append(tg.Variable(
            instance_var.value,
            role_description=instance_var.role_description
        ))

    ensembled_prediction = ensembler(predictions)
    performance_history.append(instance_eval_fn(ensembled_prediction))
    predictions.append(ensembled_prediction)

    return performance_history, predictions, question, answer

engine = "azure-gpt-35-turbo-16k"
backward_engine = "azure-gpt-4o"
num_threads = 6 
task = "MedQA"

llm_engine = tg.get_engine(engine_name=engine)
llm_backward_engine = tg.get_engine(engine_name=backward_engine)

tg.set_backward_engine(llm_backward_engine, override=True)
test_set = load_instance_task(task, evaluation_api=llm_backward_engine)

ensembler = MajorityVoting()

'''
all_solutions = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for _, sample in enumerate(test_set):
        future = executor.submit(run_test_time_training, sample)
        futures.append(future)

    all_history = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0):
        performance_history, predictions, question, answer = future.result()
        all_solutions[question] = {"predictions": [p.value for p in predictions], "answer": answer}
        all_history.append(performance_history)
'''


all_solutions = {}
all_history = []
for _, sample in enumerate(test_set):
    performance_history, predictions, question, answer = run_test_time_training(sample)
    all_solutions[question] = {"predictions": [p.value for p in predictions], "answer": answer}
    all_history.append(performance_history)



print(np.array(all_history).mean(axis=0))
with open(f"./{task}_predictions.json", "w") as f:
    json.dump(all_solutions, f)