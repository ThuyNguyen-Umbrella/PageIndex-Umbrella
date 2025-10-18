from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from transformers import pipeline
import asyncio
import torch


class LlamaModel(BaseModel):
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        super().__init__(model_name)
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Lazy-load the LLaMA model via pipeline (auto-handles mask & padding)."""
        if self.pipeline is None:
            logging.info(f"Loading LLaMA model: {self.model_name}")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

    def generate(self, prompt, chat_history=None, include_finish_reason=False):
        """
        Generate text output using the LLaMA pipeline (chat-style conversation).
        Automatically handles attention_mask and pad_token_id.
        """
        self._load_model()

        try:
            # Build chat messages
            messages = chat_history[:] if chat_history else []
            messages.append({"role": "user", "content": prompt})

            # Generate using pipeline
            outputs = self.pipeline(
                messages,
                max_new_tokens=128000,
                temperature=0.7,
                do_sample=True,
            )

            # LLaMA returns structured message output; extract the final assistant message
            generated_text = outputs[0]["generated_text"][-1]["content"].strip()

            if include_finish_reason:
                return generated_text, "finished"

            return generated_text

        except Exception as e:
            logging.error(f"Error during generation: {e}")
            return "Error"
    async def generate_async(self, prompt):
        self._load_model()
        try:
            # Build chat messages
            messages = {"role": "user", "content": prompt}

            # Generate using pipeline
            outputs = self.pipeline(
                messages,
                max_new_tokens=128000,
                temperature=0.7,
                do_sample=True,
            )

            # LLaMA returns structured message output; extract the final assistant message
            generated_text = outputs[0]["generated_text"][-1]["content"].strip()

            return generated_text

        except Exception as e:
            logging.error(f"Error during generation: {e}")
            return "Error"

if __name__ == "__main__":
    # model = LlamaModel()
    # prompt = "Explain the theory of relativity."
    # response = await model.generate_async(prompt)
    # print(response)


    model = LlamaModel()
    prompt = "Explain the theory of relativity."

    response = model.generate(prompt)
    print(response)

    # async def main():
    #     response = await model.generate_async(prompt)
    #     print(response)

    # asyncio.run(main())