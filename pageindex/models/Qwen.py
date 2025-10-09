from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

#modelname = Qwen/Qwen3-4B-Instruct-2507
#modelName = Qwen/Qwen2.5-7B-Instruct


class QwenModel(BaseModel):
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        super().__init__(model_name)
        self.client = None

    def _load_model(self):
        if self.client is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.client = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype="auto", device_map="auto"
            )

    def generate(self, prompt, chat_history=None, include_finish_reason=False):
        self._load_model()
        try:
            if chat_history:
                messages = chat_history
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            print('DEVICE:', self.client.device)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.client.device)

            # conduct text completion
            generated_ids = self.client.generate(**model_inputs, max_new_tokens=2048)
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

            if include_finish_reason:
                if len(response) >= 4000:
                    return response, "max_output_reached"
                else:
                    return response, "finished"
            output_tokens = len(self.tokenizer.encode(response))
            if include_finish_reason:
                if output_tokens >= 30000:
                    return response, "max_output_reached"
                else:
                    return response, "finished"
            else:
                return response

        except Exception as e:
            logging.error(f"Error: {e}")
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
            generated_ids = self.client.generate(**model_inputs, max_new_tokens=1024)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            return response

        except Exception as e:
            logging.error(f"Error: {e}")
            return "Error"


if __name__ == "__main__":
    model = QwenModel()
    prompt = "Explain the theory of relativity."
    response = model.generate(prompt)
    print(response)