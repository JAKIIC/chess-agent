from enum import StrEnum


class SyncMode(StrEnum):
    """User-selected visual synchronization policy."""

    STRICT_SINGLE = "strict_single"
    HUMAN_VS_AI = "human_vs_ai"
