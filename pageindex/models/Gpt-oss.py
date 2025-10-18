from .BaseModel import BaseModel
from transformers import pipeline
import logging



class GptOssModel(BaseModel):
    def __init__(self, model_name="openai/gpt-oss-20b"):
        super().__init__(model_name)
        self.client = None

    def _load_model(self):
        if self.client is None:
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.client = pipeline(
                "text-generation",
                model=self.model_name,
                dtype="auto",
                device_map="auto",
            )

    def generate(self, prompt, chat_history=None, include_finish_reason=False):
        self._load_model()
        try:
            if chat_history:
                messages = chat_history + [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

            # Generate response
            outputs = self.client(messages, max_new_tokens=1024)

            # Extract text (HF returns a list of generations)
            # Each item has "generated_text" → list of message dicts
            generated = outputs[0]["generated_text"]
            response = generated[-1]["content"]

            if include_finish_reason:
                # Hugging Face doesn’t expose finish reason here — infer by length
                if len(response) >= 1024:
                    return response, "max_output_reached"
                else:
                    return response, "finished"
            else:
                return response

        except Exception as e:
            logging.error(f"Error in generate(): {e}")
            return "Error"


    async def generate_async(self, prompt):
        self._load_model()
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.client.device)

            # conduct text completion
            generated_ids = self.client.generate(**model_inputs, max_new_tokens=32768)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            # response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            return response

        except Exception as e:
            logging.error(f"Error: {e}")
            return "Error"


if __name__ == "__main__":
    model = GptOssModel()
    prompt = "Explain the theory of relativity."
    response = model.generate(prompt)
    print(response)