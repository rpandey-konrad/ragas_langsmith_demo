from typing import Any, Dict, List, Optional
from pydantic import PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_community.chat_models.ollama import ChatOllama


class ChatOllamaWithUsage(BaseChatModel):
    model: str = "llama3"
    temperature: float = 0

    # 👇 Declare private attribute properly
    _ollama: ChatOllama = PrivateAttr()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._ollama = ChatOllama(
            model=self.model,
            temperature=self.temperature
        )

    @property
    def _llm_type(self) -> str:
        return "ollama-with-usage"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:

        response = self._ollama.invoke(messages)

        content = response.content
        meta = getattr(response, "response_metadata", {}) or {}

        prompt_tokens = meta.get("prompt_eval_count", 0)
        completion_tokens = meta.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }

        generation = ChatGeneration(
            message=AIMessage(content=content)
        )

        return ChatResult(
            generations=[generation],
            llm_output=llm_output,
        )