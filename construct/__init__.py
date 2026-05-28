"""Hypergraph construction helpers for HyperBranch."""

from importlib import import_module

__all__ = ["ConstructConfig", "construct_hypergraph", "read_contexts"]


def __getattr__(name: str):
    if name in __all__:
        builder = import_module(".builder", __name__)
        return getattr(builder, name)
    raise AttributeError(name)
