from .BaseModel import BaseModel
import openai
import logging
import time
import asyncio


class GPTModel(BaseModel):
    def __init__(self, model_name, api_key, max_retries=10, sleep_time=15):
        super().__init__(model_name)
        self.api_key = api_key
        self.client = None
        self.max_retries = max_retries
        self.sleep_time = sleep_time

    def generate(self, prompt, chat_history=None, include_finish_reason=False):
        if self.client is None:
            self.client = openai.OpenAI(api_key=self.api_key)
        for i in range(self.max_retries):
            try:
                if chat_history:
                    messages = chat_history
                    messages.append({"role": "user", "content": prompt})
                else:
                    messages = [{"role": "user", "content": prompt}]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                )
                if include_finish_reason:
                    if response.choices[0].finish_reason == "length":
                        return response.choices[0].message.content, "max_output_reached"
                    else:
                        return response.choices[0].message.content, "finished"
                else:
                    return response.choices[0].message.content

            except Exception as e:
                print("************* Retrying *************")
                logging.error(f"Error: {e}")
                if i < self.max_retries - 1:
                    print('Sleeping for', self.sleep_time)
                    time.sleep(self.sleep_time)  # Wait for 1秒 before retrying
                else:
                    logging.error("Max retries reached for prompt: " + prompt)
                    return "Error"

    async def generate_async(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        for i in range(self.max_retries):
            try:
                async with openai.AsyncOpenAI(api_key=self.api_key) as client:
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                print("************* Retrying *************")
                logging.error(f"Error: {e}")
                if i < self.max_retries - 1:
                    print('Sleeping for', self.sleep_time)
                    await asyncio.sleep(self.sleep_time)  # Wait for 1s before retrying
                else:
                    logging.error("Max retries reached for prompt: " + prompt)
                    return "Error"


if __name__ == "__main__":
    model = GPTModel(model_name="gpt-4o-2024-11-20", api_key="your_api_key_here")
    prompt = "Explain the theory of relativity."
    response = model.generate(prompt)
    print(response)