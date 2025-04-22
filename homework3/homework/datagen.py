def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Generate a dataset of prompts and completions using the LLM.
    The dataset will be saved to the specified JSON file.
    """
    from tqdm import tqdm
    import json
    from homework.cot import CoTModel
    from homework.data import is_answer_valid
    
    # Initialize the model
    model = CoTModel()

    # Load the prompts from the JSON file
    with open("data/train.json", "r") as f:
        prompts = json.load(f)

    # Generate completions for each prompt
    dataset = []
    for prompt in tqdm(prompts, desc="Generating dataset"):
        ground_truth = prompt[1]
        ground_truth = float(ground_truth)
        print('question:',prompt[0])
        print(f'\n{ground_truth = }')
        prompt = model.format_prompt(prompt[0])
        
        completion = model.batched_generate([prompt], num_return_sequences=10, temperature=temperature)
        print(f'\n{completion = }')
        # Oversample if needed
        i = 0
        correct = False
        while (i < oversample) & (not correct):

            resp = completion[i]
            if '</answer>' in resp:
              resp = resp.split('<answer>')[1]
              resp = resp.split('</answer>')[0]
              print(f'{resp = }')
              try:
                resp = float(resp)
                if is_answer_valid(resp, ground_truth):
                  correct = True
                  dataset.append({"prompt": prompt[0], "completion": resp})
              except:
                pass
            i += 1

            
    # Save the dataset to the specified JSON file
    if len(dataset) > 0:
        with open(output_json, "w") as f:
            json.dump(dataset, f, indent=4)

def is_correct(question: str, generated_answer: str) -> bool:
    from .data import Dataset, benchmark
    try:
        parts = generated_answer.split("<answer>")
        if len(parts) == 2:
            answer_str = parts[1].split("</answer>")[0].strip()
            ground_truth_dataset = Dataset("valid") # Using valid set for checking
            for q, gt_answer in ground_truth_dataset:
                if q == question:
                    try:
                        predicted_answer = float(answer_str)
                        return abs(predicted_answer - gt_answer) < 1e-3 # Tolerance for float comparison
                    except ValueError:
                        return False
            return False
        else:
            return False
    except:
        return False
    
    
if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
