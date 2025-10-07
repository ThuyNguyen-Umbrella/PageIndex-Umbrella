from .BaseModel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import asyncio


class LlamaModel(BaseModel):
    def __init__(self, model_name="meta-llama/Llama-3.1-8B"):
        super().__init__(model_name)
        self.client = None
        self.tokenizer = None

    def _load_model(self):

        if self.client is None:
            logging.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if not getattr(self.tokenizer, "chat_template", None):
                logging.warning("No chat_template found — using default template.")
                self.tokenizer.chat_template = (
                    "<|begin_of_text|>"
                    "{% for message in messages %}"
                    "{% if message['role'] == 'system' %}"
                    "<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                    "{% elif message['role'] == 'user' %}"
                    "<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                    "{% elif message['role'] == 'assistant' %}"
                    "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                    "{% endif %}"
                    "{% endfor %}"
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                )

            self.client = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            logging.info("Model loaded successfully.")

    def _prepare_prompt(self, prompt, chat_history=None):

        messages = chat_history.copy() if chat_history else []
        messages.append({"role": "user", "content": prompt})

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logging.warning(f"apply_chat_template failed: {e}. Using manual formatting.")
            text = ""
            for m in messages:
                role = m.get("role", "user")
                text += f"{role.capitalize()}: {m['content']}\n"
            text += "Assistant:"

        return text

    def generate(self, prompt, chat_history=None, include_finish_reason=False):

        self._load_model()
        try:
            text = self._prepare_prompt(prompt, chat_history)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.client.device)

            logging.debug(f"Using device: {self.client.device}")
            generated_ids = self.client.generate(
                **model_inputs,
                max_new_tokens=512,
                max_length=4096,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            if include_finish_reason:
                output_tokens = len(self.tokenizer.encode(response))
                reason = "max_output_reached" if output_tokens >= 8192 else "finished"
                return response.strip(), reason

            return response.strip()

        except Exception:
            logging.exception("Error during text generation.")
            return "Error"

    async def generate_async(self, prompt):
        self._load_model()
        try:
            text = self._prepare_prompt(prompt)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.client.device)

            logging.debug(f"Using device: {self.client.device}")
            generated_ids = self.client.generate(
                **model_inputs,
                max_new_tokens=512,
                max_length=4096,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            output_tokens = len(self.tokenizer.encode(response))
            reason = "max_output_reached" if output_tokens >= 8192 else "finished"
            return response.strip()



        except Exception:
            logging.exception("Error during text generation.")
            return "Error"

if __name__ == "__main__":
    # model = LlamaModel()
    # prompt = "Explain the theory of relativity."
    # response = await model.generate_async(prompt)
    # print(response)


    model = LlamaModel()
    prompt = "Explain the theory of relativity."

    async def main():
        response = await model.generate_async(prompt)
        print(response)

    asyncio.run(main())