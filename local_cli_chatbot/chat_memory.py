from collections import deque
from typing import List, Dict

class ChatMemory:
    """Sliding-window conversational memory.

    Stores alternating user/bot messages and can render the last N *turns* as a prompt.
    """
    def __init__(self, window_size: int = 4):
        assert window_size >= 1, "window_size must be >= 1"
        self.window_size = window_size
        self.history: List[Dict[str, str]] = []  # e.g., {'role': 'user'|'bot', 'text': '...'}

    def reset(self):
        self.history.clear()

    def add_message(self, role: str, text: str):
        role = role.lower().strip()
        assert role in {"user", "bot"}, "role must be 'user' or 'bot'"
        self.history.append({"role": role, "text": text})

    def add_user(self, text: str):
        self.add_message("user", text)

    def add_bot(self, text: str):
        self.add_message("bot", text)

    def _last_n_turns(self) -> List[Dict[str, str]]:
        """Return the last N *turns* (a turn is user+bot), flattened as messages.

        If the history starts with a user message (it should), then the last N*2 messages
        cover the most recent N turns. If fewer messages exist, return what we have.
        """
        max_msgs = self.window_size * 2
        return self.history[-max_msgs:] if len(self.history) > max_msgs else self.history[:]

    def render_context(self) -> str:
        """Format recent history to a prompt preamble (without the next user input)."""
        msgs = self._last_n_turns()
        lines = []
        
        # Add a system prompt for better conversation flow
        if not lines:
            lines.append("The following is a conversation between a helpful AI assistant and a user.")
            lines.append("The AI assistant provides helpful, accurate, and friendly responses.")
            lines.append("")
        
        for m in msgs:
            prefix = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {m['text'].strip()}")
        
        return "\n".join(lines)

    def build_prompt(self, new_user_message: str) -> str:
        """Build the full prompt ending with 'Assistant:' cue for generation."""
        context = self.render_context()
        if context:
            return f"{context}\nUser: {new_user_message.strip()}\nAssistant:"
        else:
            return f"User: {new_user_message.strip()}\nAssistant:"
