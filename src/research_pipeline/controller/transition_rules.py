from ..enums import PipelineState


def requires_split_for_transition(new_state: PipelineState) -> bool:
    return new_state == PipelineState.PARAMETER_RESEARCH

