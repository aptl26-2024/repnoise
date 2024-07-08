# %%
import ray
import os
import json
from representation_noising.loss import rep_noise_loss
from representation_noising.datasets import construct_beavertails_dataset, construct_beavertails_test_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from representation_noising.evaluation import evaluate_harmfulness
import torch
import gc
import fcntl  # For Unix-based systems

def get_evaluated_models():
    evaluated_models = set()
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    evaluated_models.add((result['model'], result['jsonl_path']))
                except json.JSONDecodeError:
                    continue
    return evaluated_models


i = 0 

# base_dir = "../../models/fted/Llama-2-7b-chat-hf/LossType.QUESTION_ANSWER/ft-skip_split0"
base_dir = "../../models/fted/repnoise_0.001_beta/LossType.QUESTION_ANSWER/ft-skip_split0"
results_file = "results.jsonl"

def get_model_dirs(base_dir):
    return [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def append_result_to_file(result):
    with open(results_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(result, f)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

    
# harmful_dataloader = accelerator.prepare(harmful_dataloader)
# harmless_dataloader = accelerator.prepare(harmless_dataloader)
jsonl_paths = [
    '../../data/beavertails/non_harm_ind_abuse_dataset.jsonl',
    '../../data/beavertails/harm_ind_dataset.jsonl',           
]

print(f"{len(get_model_dirs(base_dir))=}")
results = []

# %%

@ray.remote(num_gpus=1)
def main(model_dir, jsonl_path):
    global i
    try:
        test_dataloader = construct_beavertails_test_dataset(
            tokenizer, test_batch_size=24, jsonl_path=jsonl_path,
        )
        test_dataloader = accelerator.prepare(test_dataloader)
        print(f"Processing model: {model_dir}")
        
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model = accelerator.prepare(model)

        baseline_harmfulness_scores = evaluate_harmfulness(model, test_dataloader, tokenizer, "beavertails")
        avg_harmfulness_score = sum(baseline_harmfulness_scores[0]['scores']) / len(baseline_harmfulness_scores[0]['scores'])
        
        result = {
            "model": model_dir,
            "jsonl_path": jsonl_path,
            "harmfulness_score": avg_harmfulness_score,
            "detailed_scores": baseline_harmfulness_scores[0]['scores']
        }
        
        results.append(result)
        append_result_to_file(result)
        i += 1
        print(f"Model: {model_dir}\n{jsonl_path=}")
        print(f"harmfulness score: {avg_harmfulness_score}")
        print(f"Result appended to file. {i=}.")
        print("-" * 50)
        # print(json.dumps(result, indent=4))
        model.to("cpu")
        del model
        gc.collect()

    except Exception as e:
        error_result = {
            "model": model_dir,
            "error": str(e)
        }
        results.append(error_result)
        append_result_to_file(error_result)
        print(f"Error processing {model_dir}: {str(e)}")
        print("Error logged to file.")
        print("-" * 50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

ray.init()
evaluated_models = get_evaluated_models()
refs = []
for model_dir in get_model_dirs(base_dir):
    for jsonl_path in jsonl_paths:
        if (model_dir, jsonl_path) not in evaluated_models:
            refs += [main.remote(model_dir, jsonl_path)]
    
for ref in refs:
    ray.get(ref)

ray.shutdown()
print(f"All results have been stored in {results_file}")
