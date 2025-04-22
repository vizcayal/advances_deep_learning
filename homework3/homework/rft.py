from .base_llm import BaseLLM
from .sft import test_model
from .data import Dataset, benchmark

from transformers import AutoTokenizer, AutoModelForCausalLM


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

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

def format_example(prompt: str, answer: str, reasoning:str) -> dict[str, str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    return {"question": prompt, 
            "answer": f"<answer>{answer:.3f}</answer>",
            "reasoning": reasoning
          }

def tokenize(tokenizer, question: str, answer: str, reasoning:str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {reasoning} {answer}{tokenizer.eos_token}"

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

def train_model(
    output_dir: str = "homework/rft_model",
    learning_rate: float = 1e-3,
    num_train_epochs: int = 30,
    per_device_train_batch_size: int = 64,
):
  from transformers import Trainer, TrainingArguments
  from peft import get_peft_model, LoraConfig
  tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    
  llm = BaseLLM()

  # Configure LoRA
  config = LoraConfig(
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
    r=16,  # Adjust rank to control model size
    lora_alpha=64, # Adjust alpha based on rank
    #lora_dropout=0.1,
  )

  llm.model = get_peft_model(llm.model, config)
  llm.model.print_trainable_parameters()
  llm.model.enable_input_require_grads()

  # Prepare dataset
  train_dataset = Dataset("rft")
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

  )

  trainer = Trainer(
    model=llm.model,
    args=training_args,
    train_dataset=formatted_train_dataset,
  )

  trainer.train()
  trainer.save_model(output_dir)
  test_model(output_dir)

if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})