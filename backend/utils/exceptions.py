class InfoGatherAgentException(Exception):
    """Base exception for InfoGatherAgent."""


class LLMResponseParseError(InfoGatherAgentException):
    """LLM response parse error."""


class ToolExecutionError(InfoGatherAgentException):
    """Tool execution error."""


from backend.agent.core.exceptions import MaxIterationsExceeded

__all__ = [
    "InfoGatherAgentException",
    "LLMResponseParseError",
    "ToolExecutionError",
    "MaxIterationsExceeded",
]
