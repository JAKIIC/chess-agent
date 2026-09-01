"""Local diagnostics with privacy-preserving logging and endpoint evidence."""

from xiangqi_agent.diagnostics.endpoint_samples import (
    EndpointCrops,
    EndpointSampleRecorder,
    EndpointSampleV1,
    SampleKind,
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
    "SampleKind",
    "TransitionPointCrops",
    "TransitionSampleRecorder",
    "TransitionSampleV2",
]
