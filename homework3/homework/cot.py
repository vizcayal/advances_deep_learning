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
                      "content":"your are a converter assistant for converting units\
                                for example if you are asked to transform 1 m to cm\
                                you multiply 1 by 100 and the answer is 100 cm\
                                please be concise"
                    },
                    {
                      "role":"user",
                      "content": question
                    }
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
