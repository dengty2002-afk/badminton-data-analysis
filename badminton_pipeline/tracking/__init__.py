"""Frame enrichment, court projection, and trajectory post-processing."""

__all__ = [
    "annotate_play_state_evidence",
    "apply_play_state_gate",
    "assign_rally_ids",
    "enrich_records",
    "score_hit_candidates",
]


def __getattr__(name: str):
    if name == "assign_rally_ids":
        from .rallies import assign_rally_ids

        return assign_rally_ids
    if name == "annotate_play_state_evidence":
        from .play_state import annotate_play_state_evidence

        return annotate_play_state_evidence
    if name == "apply_play_state_gate":
        from .play_state_gate import apply_play_state_gate

        return apply_play_state_gate
    if name in __all__:
        from . import postprocess

        return getattr(postprocess, name)
    raise AttributeError(name)
