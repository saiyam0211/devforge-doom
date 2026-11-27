"""DevForge submission entry point for the trained neural policy."""

try:
    from neural_agent import DoomAgent
except ImportError:
    from submission.neural_agent import DoomAgent

__all__ = ["DoomAgent"]
