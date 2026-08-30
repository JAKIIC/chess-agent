from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xiangqi_agent.domain.board import Side

type GamePhase = Literal["opening", "middlegame", "endgame"]
type ExplanationSource = Literal["deepseek", "local_fallback"]


class CoachCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate_id: str = Field(pattern=r"^candidate_[1-9][0-9]*$")
    uci: str = Field(pattern=r"^[a-i][0-9][a-i][0-9]$")
    notation: str = Field(min_length=1)
    score_cp: int | None
    mate_in: int | None
    depth: int = Field(ge=0)
    pv_uci: tuple[str, ...] = Field(min_length=1)
    pv_notation: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _exactly_one_score(self) -> CoachCandidate:
        if (self.score_cp is None) == (self.mate_in is None):
            raise ValueError("candidate must have exactly one cp or mate score")
        if len(self.pv_uci) != len(self.pv_notation):
            raise ValueError("candidate PV coordinates and notation must align")
        return self


class ImmediateTactic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate_id: str
    is_capture: bool
    captured_piece: str | None
    gives_check: bool


class CoachEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    position_id: str = Field(min_length=1)
    fen: str = Field(min_length=1)
    user_side: Side
    phase: GamePhase
    recent_moves: tuple[str, ...]
    material_facts: dict[str, int]
    king_safety_facts: dict[str, bool]
    immediate_tactics: tuple[ImmediateTactic, ...]
    candidates: tuple[CoachCandidate, ...] = Field(min_length=1, max_length=3)
    allowed_move_map: dict[str, str]
    actual_move_review: str | None = None


class CoachExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    position_id: str = Field(min_length=1)
    position_summary: str = Field(min_length=1)
    main_plan: str = Field(min_length=1)
    candidate_id: str = Field(pattern=r"^candidate_[1-9][0-9]*$")
    why: str = Field(min_length=1)
    opponent_threat: str = Field(min_length=1)
    alternatives: tuple[str, ...]
    training_question: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    source: ExplanationSource
