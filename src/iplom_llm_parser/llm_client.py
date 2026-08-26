import asyncio
import logging
import os

import httpx2
import regex as re
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.settings import ModelSettings

from iplom_llm_parser.config import LLMConfig
from iplom_llm_parser.prompts import PROMPTS
from iplom_llm_parser.template_cor import correct_single_template

logger = logging.getLogger(__name__)


class TemplateResult(BaseModel):
    template: str


class LLMConnectionError(RuntimeError):
    """Raised when the configured LLM endpoint is unreachable, unauthenticated,
    or not serving the configured model."""


def _resolve_api_key(config: LLMConfig) -> str:
    if config.api_key:
        return config.api_key

    env_api_key = os.environ.get("API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if config.provider == "local":
        if not env_api_key:
            logger.warning(
                "API_KEY not set, depending on local LLM configuration this might fail"
            )
        return env_api_key or "local"

    if not env_api_key:
        raise RuntimeError(
            "API key not found. Set OPENROUTER_API_KEY (or API_KEY) env var or pass llm.api_key in config."
        )
    return env_api_key


def check_llm_connection(llm_config: LLMConfig, timeout: float = 5.0) -> None:
    if not llm_config.verify_connection:
        return

    try:
        api_key = _resolve_api_key(llm_config)
    except RuntimeError as e:
        raise LLMConnectionError(str(e)) from e

    if llm_config.provider == "local":
        auth_url = None
        models_url = llm_config.base_url.rstrip("/") + "/models"
        remedy = (
            f"Start a local OpenAI-compatible server (e.g. LM Studio) "
            f"serving '{llm_config.model}' at {llm_config.base_url}, or switch "
            f"provider to 'openrouter' with an API key set."
        )
    else:
        auth_url = "https://openrouter.ai/api/v1/key"
        models_url = "https://openrouter.ai/api/v1/models"
        remedy = "Check that OPENROUTER_API_KEY (or API_KEY) holds a valid key."

    headers = {"Authorization": f"Bearer {api_key}"}

    def get(url: str) -> httpx2.Response:
        try:
            return httpx2.get(url, timeout=timeout, headers=headers)
        except httpx2.HTTPError as e:
            raise LLMConnectionError(
                f"Cannot reach {url} ({type(e).__name__}). {remedy}"
            ) from e

    if auth_url:
        response = get(auth_url)
        if response.status_code != 200:
            raise LLMConnectionError(
                f"{auth_url} rejected the key with HTTP {response.status_code}. {remedy}"
            )

    response = get(models_url)
    if response.status_code != 200:
        raise LLMConnectionError(
            f"{models_url} answered HTTP {response.status_code}. {remedy}"
        )

    served = sorted(
        filter(None, (e.get("id") for e in response.json().get("data", [])))
    )
    if served and llm_config.model not in served:
        choices = (
            "See https://openrouter.ai/models for the ids."
            if len(served) > 20
            else f"Available: {', '.join(served)}"
        )
        raise LLMConnectionError(
            f"'{llm_config.model}' is not served by {models_url}. {choices}"
        )


def _build_model(config: LLMConfig) -> OpenAIChatModel | OpenRouterModel:
    api_key = _resolve_api_key(config)
    if config.provider == "local":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(base_url=config.base_url, api_key=api_key),
        )
    else:
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(
            config.model, provider=OpenRouterProvider(api_key=api_key)
        )


class LLMClient:
    def __init__(self, config: LLMConfig):
        check_llm_connection(config)
        self._agent = Agent(
            _build_model(config),
            output_type=TemplateResult,
            system_prompt="You are an expert log parser.",
            model_settings=ModelSettings(
                temperature=config.temperature,
                thinking=config.reasoning,
                extra_body={
                    "thinking": {"type": "enabled" if config.reasoning else "disabled"}
                },
            ),
        )
        self._config = config
        self._prompt = PROMPTS[config.prompt]

        self._loop = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

        self._llm_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0

    def _truncate_for_llm(self, msg: str, max_tokens: int = 40) -> str:
        tokens = msg.split(" ")
        if len(tokens) <= max_tokens:
            return msg
        logger.warning(f"Truncating a long log message: {msg[:max_tokens]}...")
        return " ".join(tokens[:max_tokens])

    async def _query(
        self,
        sample_messages: list[str],
    ) -> str:
        async with self._semaphore:
            if self._config.max_tokens_per_log is None:
                logs_text = "\n".join(sorted(sample_messages))
            else:
                truncated = [
                    self._truncate_for_llm(
                        m, max_tokens=self._config.max_tokens_per_log
                    )
                    for m in sample_messages
                ]
                logs_text = "\n".join(sorted(truncated))

            try:
                self._llm_calls += 1
                result = await asyncio.wait_for(
                    self._agent.run(self._prompt.format(logs_text)),
                    timeout=self._config.timeout,
                )

                input_tokens, output_tokens = (
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                )

                if not input_tokens:
                    logger.warning("Could not get INPUT token count")
                if not output_tokens:
                    logger.warning("Could not get OUTPUT token count")

                self._input_tokens += input_tokens or 0
                self._output_tokens += output_tokens or 0

                return result.output.template
            except TimeoutError:
                logger.error(
                    f"LLM query timed out after {self._config.timeout}s | sample: {sample_messages[0][:30]}..."
                )
                return sample_messages[0]
            except Exception as e:
                logger.error(
                    f"LLM query failed: {e} | sample count: {len(sample_messages)} | sample: {sample_messages[0][:30]}..."
                )
                return sample_messages[0]  # TODO: figure out better fallback

    async def _query_batch_async(self, sample_batches: list[list[str]]) -> list[str]:
        if not sample_batches:
            return []

        tasks = [self._query(batch) for batch in sample_batches]
        return await asyncio.gather(*tasks)

    def query_batch(self, sample_batches: list[list[str]]) -> list[str]:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        return self._loop.run_until_complete(self._query_batch_async(sample_batches))

    def close(self):
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def llm_calls(self):
        return self._llm_calls

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens


def postprocess(template: str, template_correction: bool) -> tuple[str, list[str]]:
    slot_types = re.findall(r"<(OID|LOI|OBN|TID|SID|TDA|CRS|OBA|STC|OTP)>", template)

    template = re.sub(r"<(OID|LOI|OBN|TID|SID|TDA|CRS|OBA|STC|OTP)>", "<*>", template)
    template = re.sub(r"\{[^}]+\}", "<*>", template)
    template = re.sub(r"<(?!\*>)[^>]+>", "<*>", template)
    template = re.sub(r"<(?!\*>)[^>]*>", "<*>", template)

    if template_correction:
        template = correct_single_template(template)

    return template.strip(), slot_types
