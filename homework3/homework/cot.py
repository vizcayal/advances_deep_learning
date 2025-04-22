# # Homework 3 - Well reasoned unit conversion

# In this homework, we will train language models to perform unit conversions (meters to yard to feet etc).
# We will use SmolLM2 and the huggingface library (with some more dependencies than usual).

# The homework consists of four parts:

# 1. Implement generation and batched generation in `base_llm.py`
# 2. Using in-context learning and chain of thought to perform basic unit conversions in `cot.py`
# 3. Fine-tune SmolLM2 (using LoRA) to learn to convert units better in `sft.py`
# 4. Implement a very basic RL algorithm RFT (Yuan etal. 2023, https://arxiv.org/abs/2308.01825) to fine-tune the model in `rft.py` and `dataset.py`

# Familiarize yourself with the starter code. All data ships with the starter code.

# We provide dataloaders for the text data in `data.py`.

# ## Grading Criteria

# Each part is worth 25 pts with 5 pts of extra credit for an especially performant RFT model.

# ## Generation with SmolLM2 (25 pts)

# We start by implementing the generation function of SmolLM2.
# We will generate from scratch rather than using huggingface pipelines.

# To warm up, implement a sequential version of `generate` in `base_llm.py`.
# If you feel confident, skip this and move straight to `batched_generate`.
# We already took care of loading the model and tokenizer.
# You can find some simple examples on how to use SmolLM2 here: <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B>

# Test your code with

# ```bash
# python -m homework.base_llm test
# ```

# Next, we implement a batched version `batched_generate`.
# Batching will make much better use of your GPU and likely work 10-20x faster than an unbatched version.
# The core structure of batched generation is very similar to regular generation, with one exception: All sequences that go into the transformer need to be of the same length.
# This is achieved through padding the shorter sequences in the left (aligning all sequences on the right, where generation starts).
# The transformers library will take care of padding in the `self.tokenizer` call, simply pass in a `list[str]` of prompts and use `padding=True` and return a PyTorch tensor `return_tensors="pt"`.
# Generation if almost the same between unbatched and batched versions with the only difference being that `self.model.generate` take both `input_ids` (the tokenized input) and `attention_mask` as input.
# `attention_mask` is produced by the tokenizer indicating which inputs have been padded.
# Finally, the `self.tokenizer` should decode the generated output using `batch_decode`.
# This will produce a flat `list[str]` of generations of length `num_return_sequences * len(prompts)`.
# Reshape this list if required (`num_return_sequences is not None`).

# ## In context learning (25 pts)

# Implement the `format_prompt` function in `cot.py`.
# Given a `question: str` you should create a chat dialogue that prompts the LLM to produce the correct answer.
# A chat dialogue has the following structure

# ```python
# messages: list[dict[str, str]] = [
#     {"role": role, "content": content},
#     ...
# ]
# ```

# where `role` is a string literal (`"system"`, `"user"`, or `"assistant"`), and `content` is a free-form string.
# You can use the chat dialogue to both instruct the model to perform a task in the system or user message, and provide in-context examples in a prior assistant message.
# The LLM will do best if you give it:

# - brief instructions
# - tell it to `be concise`
# - Give one good example how to solve the task

# Use the `self.tokenizer.apply_chat_template` with `add_generation_prompt=True` and `tokenize=False` to convert the chat messages into a single string following the chat-template SmolLM2 expects (including all special tokens, and the beginning of the assistant output).
# Feel free to print this output to familiarize yourself with how this works.

# Test your model with

# ```bash
# python -m homework.cot test
# ```

# You should be able to reach 0.5 accuracy and 0.85 answer_rate without too much tuning, and a good in-context example.

# ## Supervised fine-tuning (25 pts)

# We will now go and fine-tune SmolLM2 to answer questions directly.
# You should NOT use the chat template here, instead simply ask the model to complete a question with `<answer>{answer}</answer>`, where answer is the ground truth `float` answer.

# Due to file-size limitations you will not be able to submit a fully fine-tuned model, but will need to submit a LoRA adapter.
# Use the `get_peft_model` function to convert the `BaseLLM.model` into a LoRA adapted version.
# The function above takes a `LoraConfig` argument, most parameters are quite flexible.
# Our recommendation is to use:

# - `target_modules="all-linear"` this will add an adapter to all layers
# - `bias="none"` and `task_type="CAUSAL_LM"`
# - `r` rank such that the overall model size stays below 20MB
# - `lora_alpha` about 4-5 times the rank

# If you're using a GPU call `model.enable_input_require_grads()` after adding the LoRA adapter to avoid a bug with `gradient_checkpointing=True,` in the `TrainingArguments` below.

# We will use the higgingface `Trainer` to fine-tune the model.
# The trainer takes 3 arguments:

# - Our LoRA model
# - `TrainingArguments`
#   - Use `gradient_checkpointing=True` to save GPU memory
#   - Set a reasonable `learning_rate`
#   - Use `output_dir=output_dir`, `logging_dir=output_dir`, `report_to="tensorboard"` to create a
#     tensorboard log and checkpoints in `output_dir`
#   - You shouldn't have to train for more than 5 `num_train_epochs` with a `per_device_train_batch_size=32`
# - A `TokenizedDataset`. We provide significant part of the tokenization starter code here.

# Finally, call `Trainer.train` to train the model.

# Either write a script that moves the final checkpoint in the correct directory or call `Trainer.save` to write the model to the `homework/sft_model` directory.

# Train your model with

# ```bash
# python -m homework.sft train
# ```

# and make sure it can be loaded by the grader

# ```bash
# python -m homework.sft train
# ```

# ## Rejection sampling Fine-Tuning (25 pts)

# Finally, we implement a very basic RL algorithm to improve the reasoning capabilities of our LLM.
# The above SFT experiment produced straight up `<answer>...</answer>` outputs without first thinking about how to convert units.
# RFT will combine strengths of both Chain-of-Thought reasoning and SFT.

# RFT (Yuan etal. 2023, https://arxiv.org/abs/2308.01825) uses an offline procedure to create chain-of-thought-based answers.
# They start with a pre-trained LLM and in-context learning to create a new dataset of correct question / reasoning / answer tuples.
# We will implement this in `datagen.py`.
# Specifically, implement `generate_dataset` to produce 10 - 20 different completions from your `CoTModel`, then select the one with the correct answer, and add it to a dataset.
# If none of the answer is correct, ignore that data point.

# You should use the `CoTModel.batched_generate` function with `num_return_sequences > 1` and `temperature > 0` to produce a number of diverse outputs.
# Using the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model should further help you obtain better rollouts.
# In our experiments, we had a 90+% success rate in generating this dataset (success = > 1 / 10 samples answered correctly).
# Store the output in a json file in `data/rft.json`.
# Here is a sample entry

# ```json
#   [
#     "How many gram are there per 6 kg?",
#     6000.0,
#     "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
#   ],
# ```

# Modify your SFT code to train on this new data of question + reasoning pairs.

# This model will likely perform better than SFT, but might need a slightly larger LoRA adapter.
# Feel free to increase the rank as long as your total submission size is below 50Mb.

from IPython.core.inputtransformer2 import tokenize
from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        message = [
                    {
                      "role":"system",
                      "content":"You are a conversion calculator. avoid verbose explanation. "
                    },
                    {
                      "role":"user",
                      "content": "Can you change 2 hour to its equivalent in minutes?"
                    },
                    {
                      "role":"assistant",
                      "content": "1 hour = 60.00 min. 2 hour * 60.00 is <answer>120.00</answer>"
                    },
                    {
                      "role":"user",
                      "content": "Express 4 centuries as a quantity of week.?"
                    },
                    {
                      "role":"assistant",
                      "content": "1 century = 100.00 years. 1 year = 52.18 weeks. 4 centuries = 4 * 100.00 years = 400.  400 years * 52.1786 <answer>20871.44</answer>"
                    },
                    {
                      "role":"user",
                      "content": "What is the conversion of 2 hours to seconds?"
                    },
                    {
                      "role":"assistant",
                      "content": "1 hour = 3600.00 seconds. 2 hours = 2 * 3600.00 seconds = <answer>7200.00</answer>"
                    },
                    {
                      "role":"user",
                      "content": "How many gram are there per 6 kg?"
                    },
                    {
                      "role":"assistant",
                      "content": "1 kg = 1000.00 grams. 6 kgs = 6 * 1000 grams = <answer>6000.00</answer>"
                    },
                    {
                      "role":"user",
                      "content": question
                    },
                   


                  ]
        formatted_prompt = self.tokenizer.apply_chat_template(message, add_generation_prompt = True, tokenize = False )
        return formatted_prompt
    


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
