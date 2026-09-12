"""Export pipeline artifacts for downstream review tools."""

__all__ = ["build_batch_dataset", "build_demo_feed"]


def __getattr__(name: str):
    if name == "build_batch_dataset":
        from .batch_dataset import build_batch_dataset

        return build_batch_dataset
    if name == "build_demo_feed":
        from .demo_feed import build_demo_feed

        return build_demo_feed
    raise AttributeError(name)
