import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from inferencesh import BaseApp, BaseAppOutput, ContextMessage, LLMInput
from pydantic import Field, BaseModel
from typing import AsyncGenerator, List, Optional
from queue import Queue
from threading import Thread
import asyncio
import PIL
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os.path
import base64
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from contextlib import ExitStack

def strftime_now(*args, **kwargs):
    return datetime.now().strftime(**kwargs)

configs = {
    "default": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstral.gguf",
    },
    "q8": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstralQ8_0.gguf",
    },
    "q5": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstralQ5_K_M.gguf",
    },
    "q4": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstralQ4_K_M.gguf",
    },
    "q4_0": {
        "repo_id": "mistralai/Devstral-Small-2505_gguf",
        "model_filename": "devstralQ4_0.gguf",
    },
}

MAGISTRAL_JINJA_TEMPLATE = ("{{ '<bos>' }}"
        "{%- if messages[0]['role'] == 'system' -%}"
        "{%- if messages[0]['content'] is string -%}"
        "{%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}"
        "{%- else -%}"
        "{%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}"
        "{%- endif -%}"
        "{%- set loop_messages = messages[1:] -%}"
        "{%- else -%}"
        "{%- set first_user_prefix = \"\" -%}"
        "{%- set loop_messages = messages -%}"
        "{%- endif -%}"
        "{%- for message in loop_messages -%}"
        "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
        "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
        "{%- endif -%}"
        "{%- if (message['role'] == 'assistant') -%}"
        "{%- set role = \"model\" -%}"
        "{%- else -%}"
        "{%- set role = message['role'] -%}"
        "{%- endif -%}"
        "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}"
        "{%- if message['content'] is string -%}"
        "{{ message['content'] | trim }}"
        "{%- elif message['content'] is iterable -%}"
        "{%- for item in message['content'] -%}"
        "{%- if item['type'] == 'image_url' -%}"
        "{{ '<start_of_image>' }}"
        "{%- elif item['type'] == 'text' -%}"
        "{{ item['text'] | trim }}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- else -%}"
        "{{ raise_exception(\"Invalid content type\") }}"
        "{%- endif -%}"
        "{{ '<end_of_turn>\n' }}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{ '<start_of_turn>model\n' }}"
        "{%- endif -%}")


jinja_formatter = Jinja2ChatFormatter(
    MAGISTRAL_JINJA_TEMPLATE,
    eos_token="<end_of_turn>",
    bos_token="<bos>"
)

SYSTEM_PROMPT = """You are Devstral, a helpful agentic model trained by Mistral AI and using the OpenHands scaffold. You can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
* If the user asks a question, like "why is X happening", don't try to fix the problem. Just give an answer to the question.
</ROLE>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.
</CODE_QUALITY>

<VERSION_CONTROL>
* When configuring git credentials, use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.
* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.
* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.
* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.
</VERSION_CONTROL>

<PULL_REQUESTS>
* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.
* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.
* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.
</PULL_REQUESTS>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING:
   * For bug fixes: Create tests to verify issues before implementing fixes
   * For new features: Consider test-driven development when appropriate
   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure
   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies
4. IMPLEMENTATION: Make focused, minimal changes to address the problem
5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.
</PROBLEM_SOLVING_WORKFLOW>

<SECURITY>
* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.
* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.
</SECURITY>

<ENVIRONMENT_SETUP>
* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.
* If you encounter missing dependencies:
  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)
  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)
  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed
* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.
</ENVIRONMENT_SETUP>

<TROUBLESHOOTING>
* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:
  1. Step back and reflect on 5-7 different possible sources of the problem
  2. Assess the likelihood of each possible cause
  3. Methodically address the most likely causes, starting with the highest probability
  4. Document your reasoning process
* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.
</TROUBLESHOOTING>"""


class AppInput(LLMInput):
    system_prompt: str = Field(
        description="The system prompt to use for the model",
        default=SYSTEM_PROMPT,
        examples=[]
    )
    context: list[ContextMessage] = Field(
        description="The context to use for the model",
        examples=[
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the weather like today?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "I apologize, but I don't have access to real-time weather information. You would need to check a weather service or app to get current weather conditions for your location."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Can you help me write a poem about spring?"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "Here's a short poem about spring:\n\nGreen buds awakening,\nSoft rain gently falling down,\nNew life springs anew.\n\nWarm sun breaks through clouds,\nBirds return with joyful song,\nNature's sweet rebirth."}]}
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Explain quantum computing in simple terms"}]}, 
                {"role": "assistant", "content": [{"type": "text", "text": "Quantum computing is like having a super-powerful calculator that can solve many problems at once instead of one at a time. While regular computers use bits (0s and 1s), quantum computers use quantum bits or \"qubits\" that can be both 0 and 1 at the same time - kind of like being in two places at once! This allows them to process huge amounts of information much faster than regular computers for certain types of problems."}]}
            ]
        ],
        default=[]
    )
    temperature: float = Field(
        description="The temperature to use for the model",
        default=0.7
    )
    top_p: float = Field(
        description="The top-p to use for the model",
        default=0.95
    )
    max_tokens: int = Field(
        description="The maximum number of tokens to generate",
        default=40960
    )
    context_size: int = Field(
        description="The maximum number of tokens to use for the context (changing this will cause a model re-setup)",
        min_value=4096,
        max_value=49152,
        default=4096,
    )


class AppOutput(BaseAppOutput):
    response: str
    thinking_content: Optional[str] = None
    
def MessageBuilder(input_data: AppInput):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": input_data.system_prompt}],
        }
    ]

    # Add context messages
    for msg in input_data.context:
        message_content = []
        if msg.text:
            message_content.append({"type": "text", "text": msg.text})
        messages.append({
            "role": msg.role,
            "content": message_content
        })

    # Add user message with text and image if provided
    user_content = []
    user_text = input_data.text
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": user_content})

    return messages

def stream_generate(
    model,
    messages,
    AppOutput,
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
):
    """Stream model output, splitting <think> ... </think> and handling tags that span chunks."""

    TAG_OPEN = "<think>"
    TAG_CLOSE = "</think>"

    buffer = ""               # unparsed running buffer
    thinking_content = ""
    response_content = ""
    seen_think = False
    finished_think = False

    last_think_len = 0
    last_resp_len = 0

    def _clean(txt: str) -> str:
        return (
            txt.replace("<|im_start|>", "")
               .replace("<|im_end|>", "")
               .replace("<start_of_turn>", "")
               .replace("<end_of_turn>", "")
        )

    for chunk in model.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    ):
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if not delta:
            continue

        response_content += delta
        
        yield AppOutput(
            response=response_content,
            thinking_content="",
        )

class App(BaseApp):
    def __init__(self):
        super().__init__()
        self.last_context_size = None

    async def setup(self, metadata, context_size=None):
        self.variant_config = configs[metadata.app_variant]
        # Use context_size from input if provided, else default
        n_ctx = context_size if context_size is not None else 4096
        self.last_context_size = n_ctx
        try:
            # Check if model file is available locally
            try:
                local_path = hf_hub_download(
                    repo_id=self.variant_config["repo_id"],
                    filename=self.variant_config["model_filename"],
                    local_files_only=True
                )
                print(f"Model is already available locally at: {local_path}")
                model_is_available = True
            except Exception:
                print("Model file not found locally, will be downloaded by Llama.from_pretrained.")
                model_is_available = False

            if model_is_available:
                print("Loading previously downloaded model from cache...")
            else:
                print("Downloading and initializing Devstral model...")

            self.model = Llama.from_pretrained(
                repo_id=self.variant_config["repo_id"],
                filename=self.variant_config["model_filename"],
                verbose=True,
                n_gpu_layers=-1,
                n_ctx=n_ctx,
                local_files_only=model_is_available,
                chat_handler=jinja_formatter.to_chat_handler()
            )
            print("Model initialization complete!")
        except Exception as e:
            print(f"Error during setup: {e}")
            raise

    async def run(self, input_data: AppInput, metadata) -> AsyncGenerator[AppOutput, None]:
        # If context_size changed, re-setup the model
        if not hasattr(self, 'last_context_size') or input_data.context_size != self.last_context_size:
            print(f"Context size changed (was {getattr(self, 'last_context_size', None)}, now {input_data.context_size}), triggering re-setup.")
            await self.setup(metadata, context_size=input_data.context_size)

        # Build messages using the new AppInput fields
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": input_data.system_prompt}],
            }
        ]
        # Add context messages
        for msg in input_data.context:
            messages.append(msg)
        # Add user message
        user_content = []
        if input_data.text:
            user_content.append({"type": "text", "text": input_data.text})
        messages.append({"role": "user", "content": user_content})

        # Stream generate with user-specified parameters
        for output in stream_generate(
            self.model,
            messages,
            AppOutput,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
            max_tokens=input_data.max_tokens
        ):
            yield output

    async def unload(self):
        del self.model