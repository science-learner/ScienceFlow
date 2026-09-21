"""ScienceFlow configuration adapter over the InquiryCraft tool contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

from inquirycraft.tools import BaseTool as RuntimeBaseTool
from pydantic import BaseModel, ConfigDict


class BaseTool(RuntimeBaseTool, BaseModel):
    """Pydantic-configured tool preserving the historical ScienceFlow schema."""

    name: str
    description: str
    parameters: Optional[dict] = {}
    func_signature: Optional[str] = "()"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def input_schema(self) -> dict:
        return self.parameters or {}

    async def __call__(self, **kwargs: Any) -> Any:
        return await self.execute(**kwargs)

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_param(self) -> dict[str, Any]:
        return self.to_openai()

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute keyword arguments decoded from a model tool call."""

    @staticmethod
    def codeact_func(**kwargs: Any) -> Any:
        return None


__all__ = ["BaseTool"]
