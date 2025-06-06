"""
base_prompter.py

Abstract class definition of a multi-turn prompt builder for ensuring consistent formatting for chat-based LLMs.
"""
import sys
from abc import ABC, abstractmethod
from typing import Optional


class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family

        # Only some models define a system prompt => let subclasses handle this logic!
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None: ...

    @abstractmethod
    def get_prompt(self) -> str: ...


class PurePromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        # TODO (siddk) =>> Can't always assume LlamaTokenizer --> FIX ME!
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1
        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()


# class MegamaPromptBuilder(PromptBuilder):
#     def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
#         super().__init__(model_family, system_prompt)

#         # TODO (siddk) =>> Can't always assume LlamaTokenizer --> FIX ME!
#         self.bos, self.eos = "<s>", "</s>"

#         # Get role-specific "wrap" functions
#         self.wrap_human = lambda msg: f"In: {msg}\nOut: "
#         self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

#         # === `self.prompt` gets built up over multiple turns ===
#         self.prompt, self.turn_count = "", 0

#     def add_turn(self, role: str, message: str) -> str:
#         assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

#         if (self.turn_count % 2) == 0:
#             human_message = self.wrap_human(message)
#             wrapped_message = human_message
#         else:
#             gpt_message = self.wrap_gpt(message)
#             wrapped_message = gpt_message

#         # Update Prompt
#         self.prompt += wrapped_message

#         # Bump Turn Counter
#         self.turn_count += 1
#         # Return "wrapped_message" (effective string added to context)
#         return wrapped_message

#     def get_potential_prompt(self, message: str) -> None:
#         # Assumes that it's always the user's (human's) turn!
#         prompt_copy = str(self.prompt)

#         human_message = self.wrap_human(message)
#         prompt_copy += human_message

#         return prompt_copy.removeprefix(self.bos).rstrip()

#     def get_prompt(self) -> str:
#         # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
#         return self.prompt.removeprefix(self.bos).rstrip()
    
SYS_PROMPTS = {
    "prismatic": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
    "openvla": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
    "qwenvl": (
        "You are a helpful assistant. "
    ),
}


def format_system_prompt(system_prompt: str) -> str:
    return f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt.strip()}<|eot_id|>"

def format_qwenvl_system_prompt(system_prompt: str) -> str:
    return f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"

def format_internvl_system_prompt(system_prompt: str) -> str:
    return f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
    
class MegamaPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = format_system_prompt(
            SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        )

        # LLaMa-2 Specific
        self.bos, self.eos = "<|begin_of_text|>", "<|eot_id|>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"<|start_header_id|>human<|end_header_id|>\n\n{msg}<|eot_id|>"
        # self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"
        # NOTE: \n\n -> token id 271
        self.wrap_gpt = lambda msg: f"<|start_header_id|>gpt<|end_header_id|>\n\n{msg}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.system_prompt+self.wrap_human(message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1
        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()

class LlavaOVPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = format_system_prompt(
            SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        )

        # LLaMa-2 Specific
        self.bos, self.eos = "<|begin_of_text|>", "<|eot_id|>"

        # Get role-specific "wrap" functions
        #self.wrap_human = lambda msg: f"<|start_header_id|>human<|end_header_id|>\n\n{msg}<|eot_id|>"

        self.wrap_human = lambda msg: f"<|im_start|>user {msg}<|im_end|>"

        # self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"
        # NOTE: \n\n -> token id 271
        self.wrap_gpt = lambda msg: f"<|im_start|>assistant {msg}<|im_end|>"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            #sys_message = self.system_prompt+self.wrap_human(message)
            sys_message = self.wrap_human(message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1
        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()

class QwenVLPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = format_qwenvl_system_prompt(
            SYS_PROMPTS['qwenvl'] if system_prompt is None else system_prompt
        )

        # LLaMa-2 Specific
        self.bos, self.eos = "<|begin_of_text|>", "<|eot_id|>"

        # Get role-specific "wrap" functions
        #self.wrap_human = lambda msg: f"<|start_header_id|>human<|end_header_id|>\n\n{msg}<|eot_id|>"

        self.wrap_human = lambda msg: f"<|im_start|>user {msg}<|im_end|>"

        # self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"
        # NOTE: \n\n -> token id 271
        self.wrap_gpt = lambda msg: f"<|im_start|>assistant {msg}<|im_end|>"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            #sys_message = self.system_prompt+self.wrap_human(message)
            sys_message = self.wrap_human(message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1
        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.system_prompt + self.prompt.removeprefix(self.bos).rstrip()

class InternVLPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        internvl_system_prompt = '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'
        self.system_prompt = format_internvl_system_prompt(
            internvl_system_prompt
        )

        # LLaMa-2 Specific
        self.bos, self.eos = "<|begin_of_text|>", "<|eot_id|>"

        # Get role-specific "wrap" functions
        #self.wrap_human = lambda msg: f"<|start_header_id|>human<|end_header_id|>\n\n{msg}<|eot_id|>"

        self.wrap_human = lambda msg: f"<|im_start|>user\n{msg}<|im_end|>"

        # self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"
        # NOTE: \n\n -> token id 271
        self.wrap_gpt = lambda msg: f"<|im_start|>assistant\n{msg}<|im_end|>"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            #sys_message = self.system_prompt+self.wrap_human(message)
            sys_message = self.wrap_human(message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1
        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.system_prompt + self.prompt.removeprefix(self.bos).rstrip()

class IdeficsPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = format_system_prompt(
            SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        )

        # LLaMa-2 Specific
        self.bos, self.eos = "<|begin_of_text|>", "<|eot_id|>"

        # Get role-specific "wrap" functions
        #self.wrap_human = lambda msg: f"<|start_header_id|>human<|end_header_id|>\n\n{msg}<|eot_id|>"

        self.wrap_human = lambda msg: f"<|begin_of_text|>User:{msg}<end_of_utterance>"

        # self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"
        # NOTE: \n\n -> token id 271
        self.wrap_gpt = lambda msg: f"Assistant: {msg}<end_of_utterance>"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            #sys_message = self.system_prompt+self.wrap_human(message)
            sys_message = self.wrap_human(message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1
        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()