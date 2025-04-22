def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.5):
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
        print('\nquestion:',prompt[0])
        print(f'\n{ground_truth = }')
        question = prompt[0]
        prompt = model.format_prompt(prompt[0])
        
        completion = model.batched_generate([prompt], num_return_sequences=10, temperature=temperature)
        # Oversample if needed
        i = 0
        correct = 1
        while (i < oversample) & (correct<3):
            print(f'\n{completion[i] = }')
        
            resp = completion[i]
            if '</answer>' in resp:
              reasoning = resp.split('</answer>')[0] + '</answer>'
              resp = resp.split('<answer>')
              resp = resp[1]
              resp = resp.split('</answer>')[0]
              print(f'{reasoning = }')
              print(f'{resp = }')
              try:
                resp = float(resp)
                if is_answer_valid(resp, ground_truth):
                  correct += 1
                  dataset.append([question, resp, reasoning])
              except:
                pass
            i += 1

            
    # Save the dataset to the specified JSON file
    if len(dataset) > 0:
        with open(output_json, "w") as f:
            json.dump(dataset, f, indent=4)


    
    
if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
