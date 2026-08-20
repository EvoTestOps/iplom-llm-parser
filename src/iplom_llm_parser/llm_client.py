import asyncio
import logging
import os

import regex as re
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from iplom_llm_parser.config import LLMConfig
from iplom_llm_parser.template_cor import correct_single_template

logger = logging.getLogger(__name__)


DEFAULT_PROMPT = (
    "Given log messages below, extract a single common template.\n"
    "Replace each dynamic value with a typed placeholder based on its category.\n"
    "Keep static parts, punctuation and whitespace exactly as-is. There might be no variables in the logs.\n\n"
    "# Placeholder types:\n"
    "  <OID> - Session IDs, user IDs, etc.\n"
    "  <LOI> - Paths, URIs, IP addresses\n"
    "  <OBN> - Object names, task names, job names\n"
    "  <TID> - Type indicators\n"
    "  <SID> - Numerical switch/flag indicators\n"
    "  <TDA> - Timestamps, durations\n"
    "  <CRS> - Memory, disk space, byte counts\n"
    "  <OBA> - Counts of objects (errors, nodes, etc.)\n"
    "  <STC> - Numerical error/status codes\n"
    "  <OTP> - Any other dynamic value\n\n"
    "# Example:\n"
    "  Logs:\n"
    "    `Connecting to 192.168.1.1:9000 as user admin`\n"
    "    `Connecting to 10.0.0.5:8080 as user root`\n"
    "  Template: `Connecting to <LOI> as user <OID>`\n\n"
    "Logs:\n"
    "{}\n\n"
    "Template:"
)

NO_EXAMPLE_PROMPT = (
    "Given log messages below, extract a single common template.\n"
    "Replace each dynamic value with a typed placeholder based on its category.\n"
    "Keep static parts, punctuation and whitespace exactly as-is. There might be no variables in the logs.\n\n"
    "# Placeholder types:\n"
    "  <OID> - Session IDs, user IDs, etc.\n"
    "  <LOI> - Paths, URIs, IP addresses\n"
    "  <OBN> - Object names, task names, job names\n"
    "  <TID> - Type indicators\n"
    "  <SID> - Numerical switch/flag indicators\n"
    "  <TDA> - Timestamps, durations\n"
    "  <CRS> - Memory, disk space, byte counts\n"
    "  <OBA> - Counts of objects (errors, nodes, etc.)\n"
    "  <STC> - Numerical error/status codes\n"
    "  <OTP> - Any other dynamic value\n\n"
    "Logs:\n"
    "{}\n\n"
    "Template:"
)

SIMPLE_PROMPT = (
    "Given log messages below, extract a single common template.\n"
    "Replace each dynamic variable with <*>.\n"
    "Keep static parts, punctuation and whitespace exactly as-is. There might be no variables in the logs.\n\n"
    "Logs:\n"
    "{}\n\n"
    "Template:"
)

PROMPTS: dict[str, str] = {
    "default": DEFAULT_PROMPT,
    "no_example": NO_EXAMPLE_PROMPT,
    "simple": SIMPLE_PROMPT,
}


class TemplateResult(BaseModel):
    template: str


def _build_model(config: LLMConfig):
    if config.provider == "local":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        api_key = os.environ.get("API_KEY", "local")
        if not api_key:
            logger.warning(
                "API_KEY not set, depending on local LLM configuration this might fail"
            )
        return OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(base_url=config.base_url, api_key=api_key),
        )
    else:
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise RuntimeError("API_KEY environment variable is not set")
        return OpenRouterModel(
            config.model,
            provider=OpenRouterProvider(api_key=api_key),
        )


class LLMClient:
    def __init__(self, config: LLMConfig):
        # Model settings try to turn thinking off but depending on the model
        #  it might be silently ignored
        self._agent = Agent(
            _build_model(config),
            output_type=TemplateResult,
            system_prompt="You are an expert log parser.",
            model_settings=ModelSettings(
                temperature=0,
                thinking=False,
                extra_body={"thinking": {"type": "disabled"}},
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
            except asyncio.TimeoutError:
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
    slot_types = re.findall(r"<(OID|LOI|OBN|SID|TDA|CRS|OBA|STC|OTP)>", template)

    template = re.sub(r"<(OID|LOI|OBN|SID|TDA|CRS|OBA|STC|OTP)>", "<*>", template)
    template = re.sub(r"\{[^}]+\}", "<*>", template)
    template = re.sub(r"<(?!\*>)[^>]+>", "<*>", template)
    template = re.sub(r"<(?!\*>)[^>]*>", "<*>", template)

    if template_correction:
        template = correct_single_template(template)

    return template.strip(), slot_types
