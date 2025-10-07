from .BaseModel import BaseModel
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import logging

class GPT2Model(BaseModel):
    def __init__(self, model_name='gpt2'):
        super().__init__(model_name)
        self.client = None
    
    def _load_model(self):
        if self.client is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.client = GPT2LMHeadModel.from_pretrained('gpt2')
    def generate(self, prompt, chat_history=None, include_finish_reason=False):
        self._load_model()
        try:
            if chat_history:
                messages = chat_history
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]

            # text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True,
            # )

            print('DEVICE:', self.client.device)
            model_inputs = self.tokenizer(prompt, return_tensors="pt")

            # conduct text completion
            generated_ids = self.client.generate(**model_inputs, max_new_tokens=8192, repetition_penalty=1.2, pad_token_id=self.tokenizer.eos_token_id)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            # response = self.client(**model_inputs)
            output_tokens = len(self.tokenizer.encode(response))
            if include_finish_reason:
                if output_tokens >= 8192:
                    return response, "max_output_reached"
                else:
                    return response, "finished"
            else:
                return response

        except Exception as e:
            logging.error(f"Error: {e}")
            return "Error"
        
    def generate_async(self, prompt):
        self._load_model()
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

            # conduct text completion
            generated_ids = self.client.generate(**model_inputs, max_new_tokens=8192)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            return response

        except Exception as e:
            logging.error(f"Error: {e}")
            return "Error"


if __name__ == "__main__":
    model = GPT2Model()
    prompt = "what is ChatGPT?"
    response = model.generate(prompt)
    print(response)

