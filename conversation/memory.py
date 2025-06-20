class ConversationMemory:
    def __init__(self, max_turns=5):
        self.memory = []
        self.max_turns = max_turns

    def append_user_input(self, user_input):
        self.memory.append({"role": "user", "content": user_input})
        self._trim()

    def append_bot_response(self, bot_response):
        self.memory.append({"role": "assistant", "content": bot_response})
        self._trim()

    def _trim(self):
        if len(self.memory) > self.max_turns * 2:
            self.memory = self.memory[-self.max_turns * 2:]

    def format_history(self):
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.memory])