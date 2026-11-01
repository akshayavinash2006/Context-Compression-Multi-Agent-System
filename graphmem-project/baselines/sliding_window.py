from baselines.base import MemoryBackend


class SlidingWindow(MemoryBackend):
    """Last N messages."""

    name = "window"

    def __init__(self, cfg, counter=None, n: int | None = None):
        super().__init__(cfg, counter)
        self.n = n or cfg.experiment.sliding_window_n

    def select(self, agent, query, token_budget):
        return self.messages[-self.n:]
