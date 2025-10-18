from .BaseModel import BaseModel
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
import logging

class MistralModel(BaseModel):
    def __init__(self, model_name="mistralai/Voxtral-Mini-3B-2507"):
        super().__init__(model_name)
        self.client = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self.client is None:
            self.tokenizer = AutoProcessor.from_pretrained(self.model_name)
            self.client = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name, dtype="auto", device_map=self.device
            )

    def generate(self, prompt, chat_history=None, include_finish_reason=False):
        """Generate text output using Voxtral (chat style)."""
        self._load_model()

        try:
            # Build chat messages in Voxtral format
            messages = chat_history[:] if chat_history else []
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            })

            # Apply chat template (Voxtral expects conversation list)
            inputs = self.tokenizer.apply_chat_template(messages)
            inputs = inputs.to(self.device, dtype=torch.bfloat16)

            # Generate response
            outputs = self.client.generate(
                **inputs,
                max_new_tokens=32000,
                temperature=0.7,
                do_sample=True
            )

            # Decode new tokens (skip input prompt)
            decoded_outputs = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            response = decoded_outputs[0].strip()

            # Handle finish reason
            if include_finish_reason:
                if len(response) >= 32000:
                    return response, "max_output_reached"
                else:
                    return response, "finished"

            return response

        except Exception as e:
            logging.error(f"Error during generation: {e}")
            return "Error"

    async def generate_async(self, prompt):
        self._load_model()
        try:
            messages = [{"role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]}]

            # Apply chat template (Voxtral expects conversation list)
            inputs = self.tokenizer.apply_chat_template(messages)
            inputs = inputs.to(self.device, dtype=torch.bfloat16)

            # Generate response
            outputs = self.client.generate(
                **inputs,
                max_new_tokens=32000,
                temperature=0.7,
                do_sample=True
            )

            # Decode new tokens (skip input prompt)
            decoded_outputs = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            response = decoded_outputs[0].strip()

            return response

        except Exception as e:
            logging.error(f"Error: {e}")
            return "Error"


if __name__ == "__main__":
    model = MistralModel()
    prompt = "Explain the theory of relativity."
    response = model.generate(prompt)
    print(response)