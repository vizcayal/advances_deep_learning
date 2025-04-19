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
                      "content": "1 hour = 60 min. 2 hour * 60 is <answer>120.0<answer>"
                    },
                    {
                      "role":"user",
                      "content": "Express 4 centuries as a quantity of week.?"
                    },
                    {
                      "role":"assistant",
                      "content": "1 century = 100 years. 1 year = 52.18 weeks. 4 centuries = 4 * 100 years = 400. 400 years * 52.1786 <answer>20871.44<answer>"
                    },
                    {
                      "role":"user",
                      "content": "What is the conversion of 2 hour to seconds?"
                    },
                    {
                      "role":"assistant",
                      "content": "1 hour = 3600 seconds. 2 hours = 2 * 3600 years = <answer>7200<answer>"
                    },
                    {
                      "role":"user",
                      "content": "How many gram are there per 6 kg?"
                    },
                    {
                      "role":"assistant",
                      "content": "1 kg = 1000 grams. 6 kgs = 6 * 1000 years = <answer>6000<answer>"
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
