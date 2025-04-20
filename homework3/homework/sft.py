## Supervised fine-tuning (25 pts)

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


from .base_llm import BaseLLM
from .data import Dataset, benchmark

from transformers import AutoTokenizer, AutoModelForCausalLM


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM("HuggingFaceTB/SmolLM2-360M-Instruct")
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    return {"question": prompt, "answer": f"<answer>{float(answer):.3f}</answer>"}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)
    


def train_model(
    output_dir: str = "homework/sft_model",
    learning_rate: float = 2e-4,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 32,
):
  from transformers import Trainer, TrainingArguments
  from peft import get_peft_model, LoraConfig
  llm = BaseLLM()

  # Configure LoRA
  config = LoraConfig(
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
    r=8,  # Adjust rank to control model size
    lora_alpha=32, # Adjust alpha based on rank
    lora_dropout=0.05,
  )

  llm.model = get_peft_model(llm.model, config)
  llm.model.print_trainable_parameters()

  #if torch.cuda.is_available():
  llm.model.enable_input_require_grads()

  # Prepare dataset
  train_dataset = Dataset("train")
  formatted_train_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)

  training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    logging_dir=output_dir,
    report_to="tensorboard",
    save_strategy="epoch",
    save_total_limit=1,
    #fp16=torch.cuda.is_available(),
  )

  trainer = Trainer(
    model=llm.model,
    args=training_args,
    train_dataset=formatted_train_dataset,
  )

  trainer.train()
  trainer.save_model(output_dir)
  test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
