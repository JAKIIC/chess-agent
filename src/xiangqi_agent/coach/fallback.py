from __future__ import annotations

from xiangqi_agent.domain.coach import CoachEvidence, CoachExplanation

_PHASE_TEXT = {
    "opening": "当前属于开局阶段，重点是协调子力并争夺关键线路。",
    "middlegame": "当前属于中局阶段，重点是比较强制手段和双方王区安全。",
    "endgame": "当前属于残局阶段，重点是计算交换后的兵形与将帅活动空间。",
}


def local_explanation(evidence: CoachEvidence, _question: str) -> CoachExplanation:
    candidate = evidence.candidates[0]
    tactic = evidence.immediate_tactics[0]
    score = _score_text(candidate.score_cp, candidate.mate_in)
    facts = [
        f"{candidate.candidate_id} 对应 {candidate.notation}",
        f"Pikafish 在深度 {candidate.depth} 将它列为第一候选，红方视角评分为 {score}",
    ]
    if tactic.is_capture:
        facts.append("这一步会立即吃子")
    if tactic.gives_check:
        facts.append("这一步会立即将军")
    threat = (
        f"证据变化线中的下一手是 {candidate.pv_notation[1]}。"
        if len(candidate.pv_notation) > 1
        else "当前证据没有提供足够长的变化线来确认对手下一手。"
    )
    alternatives = tuple(item.candidate_id for item in evidence.candidates[1:])
    return CoachExplanation(
        position_id=evidence.position_id,
        position_summary=_PHASE_TEXT[evidence.phase],
        main_plan="先理解第一候选的目的，再比较其他候选造成的评分差异。",
        candidate_id=candidate.candidate_id,
        why="；".join(facts) + "。",
        opponent_threat=threat,
        alternatives=alternatives,
        training_question=f"如果不走 {candidate.notation}，你最担心局面的哪一处变化？",
        confidence=0.75,
        source="local_fallback",
    )


def _score_text(score_cp: int | None, mate_in: int | None) -> str:
    if score_cp is not None:
        return f"{score_cp / 100:+.2f}"
    if mate_in is None:
        return "未知"
    return f"{'红方' if mate_in > 0 else '黑方'}杀{abs(mate_in)}步"
