"""Logical clock: one agent message = one turn."""


class Clock:
    def __init__(self, start: int = 0):
        self.now = start

    def tick(self) -> int:
        self.now += 1
        return self.now

    def __int__(self) -> int:
        return self.now

    def __mod__(self, n: int) -> int:
        return self.now % n
