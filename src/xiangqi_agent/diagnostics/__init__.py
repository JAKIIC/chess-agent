"""Local diagnostics with privacy-preserving logging and endpoint evidence."""

from xiangqi_agent.diagnostics.endpoint_samples import (
    EndpointCrops,
    EndpointSampleRecorder,
    EndpointSampleV1,
    SampleKind,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    HumanAiStageCSampleRecorder,
    HumanAiStageCSampleV1,
    StageCCandidateRecord,
    StageCExpectedOutcome,
    StageCObservedStatus,
    StageCScenario,
)
from xiangqi_agent.diagnostics.transition_samples import (
    TransitionPointCrops,
    TransitionSampleRecorder,
    TransitionSampleV2,
)

__all__ = [
    "EndpointCrops",
    "EndpointSampleRecorder",
    "EndpointSampleV1",
    "HumanAiStageCSampleRecorder",
    "HumanAiStageCSampleV1",
    "SampleKind",
    "StageCCandidateRecord",
    "StageCExpectedOutcome",
    "StageCObservedStatus",
    "StageCScenario",
    "TransitionPointCrops",
    "TransitionSampleRecorder",
    "TransitionSampleV2",
]
