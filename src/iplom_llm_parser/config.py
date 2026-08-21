import logging
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    content_col: str = "Content"
    chunk_size: int = 5000
    llm_sample_n: int = 10
    repool_passes: int = 3
    template_correction: bool = True
    infer_slot_regexes: bool = True
    singleton_fallback: bool = True


class LLMConfig(BaseModel):
    provider: Literal["local", "openrouter"]
    model: str
    base_url: str | None = None
    max_concurrent: int = 4
    timeout: int = 60
    max_tokens_per_log: int | None = None
    prompt: Literal["default", "no_example", "simple"] = "default"
    reasoning: bool = False
    temperature: float = 0.0
    api_key: str | None = None

    @field_validator("max_tokens_per_log")
    @classmethod
    def _zero_off(cls, v: int | None) -> int | None:
        return None if v == 0 else v

    @model_validator(mode="after")
    def _check_base_url(self) -> "LLMConfig":
        if self.provider == "local" and not self.base_url:
            raise ValueError("llm.base_url is required when llm.provider = 'local'")
        return self


class IPLoMConfig(BaseModel):
    CT: float = 0.3
    lower_bound: float = 0.25


class Config(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    iplom: IPLoMConfig = Field(default_factory=IPLoMConfig)
    llm: LLMConfig

    @model_validator(mode="after")
    def _simple_prompt_disables_slot_regexes(self) -> "Config":
        if self.llm.prompt == "simple" and self.pipeline.infer_slot_regexes:
            logger.warning(
                "llm.prompt = 'simple' does not produce typed slots, forcing pipeline.infer_slot_regexes to False."
            )
            self.pipeline.infer_slot_regexes = False
        return self


def load_config(path: str = "config.toml") -> Config:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Config file '{path}' not found")

    with file.open("rb") as f:
        raw = tomllib.load(f)

    try:
        return Config.model_validate(raw)
    except Exception as e:
        raise ValueError(f"Invalid config in '{path}': {e}") from e
