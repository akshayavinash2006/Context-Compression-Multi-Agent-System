from baselines.base import MemoryBackend


class FullHistory(MemoryBackend):
    """Every message so far, in order (normal practice; main baseline)."""

    name = "full"

    def select(self, agent, query, token_budget):
        return list(self.messages)
