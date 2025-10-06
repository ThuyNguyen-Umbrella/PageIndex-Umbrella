class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt, chat_history=None):
        pass

    async def generate_async(self, prompt):
        pass

