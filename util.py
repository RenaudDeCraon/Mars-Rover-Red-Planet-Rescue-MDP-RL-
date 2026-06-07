"""
util.py -- shared utilities for the Mars Rover RL project (Python 3.10+).

Exports Counter (a dict subclass with 0-default lookup, arithmetic,
and scalar-or-dot ``__mul__``), the stochastic helpers
``flip_coin`` / ``sample`` / ``sample_from_counter``, and
``raise_not_defined`` for student stubs (the caller's file/line/
method are baked into the NotImplementedError message so the
autograder can surface them without touching stderr).
"""

import inspect
import random
import sys


class Counter(dict):
    """Dict subclass whose missing keys read as 0."""

    def __getitem__(self, key):
        # dict.get returns the default without mutating -- peeking at
        # a missing key must not silently insert it.
        return dict.get(self, key, 0)

    def arg_max(self):
        """Return the key with the largest value, or None if empty."""
        if not self:
            return None
        best_key, best_val = None, None
        for key, val in self.items():
            if best_val is None or val > best_val:
                best_key, best_val = key, val
        return best_key

    def sorted_keys(self):
        """Return keys sorted by value in descending order."""
        return [k for k, _ in sorted(self.items(), key=lambda kv: kv[1], reverse=True)]

    def total_count(self):
        """Return the sum of all stored values."""
        return sum(self.values())

    def normalize(self):
        """Scale values in place so they sum to 1 (no-op on zero total)."""
        total = float(self.total_count())
        if total == 0:
            return
        for k in list(self.keys()):
            self[k] = self[k] / total

    def divide_all(self, divisor):
        """Divide every stored value by ``divisor`` in place."""
        divisor = float(divisor)
        for k in list(self.keys()):
            self[k] = self[k] / divisor

    def copy(self):
        """Return a shallow copy as another ``Counter``."""
        return Counter(dict.copy(self))

    def __mul__(self, other):
        """Scalar-multiply this counter, or dot-product it with another.

        Counter * Counter iterates over the shorter side and returns
        the scalar dot product. Counter * scalar returns a new
        Counter with every value multiplied.
        """
        if isinstance(other, Counter):
            smaller, larger = (self, other) if len(self) < len(other) else (other, self)
            return sum(v * larger[k] for k, v in smaller.items() if k in larger)
        result = Counter()
        for k, v in self.items():
            result[k] = v * other
        return result

    def __rmul__(self, other):
        """Support ``scalar * counter`` in addition to ``counter * scalar``."""
        return self.__mul__(other)

    def __add__(self, other):
        """Element-wise sum over the union of keys."""
        result = Counter()
        for k, v in self.items():
            result[k] = v + other[k]
        for k, v in other.items():
            if k not in self:
                result[k] = v
        return result

    def __sub__(self, other):
        """Element-wise difference over the union of keys."""
        result = Counter()
        for k, v in self.items():
            result[k] = v - other[k]
        for k, v in other.items():
            if k not in self:
                result[k] = -v
        return result

    def __radd__(self, other):
        """Support ``sum([counter1, counter2, ...])`` which starts at 0."""
        if other == 0:
            return self.copy()
        return self.__add__(other)


def flip_coin(p):
    """Return True with probability ``p`` and False otherwise."""
    return random.random() < p


def sample(distribution, values=None):
    """Draw a single sample from a discrete distribution.

    Two call styles:

    * ``sample(counter)`` -- delegates to :func:`sample_from_counter`.
    * ``sample(probs, values)`` -- parallel sequences; probs are
      auto-normalised before sampling via inverse-CDF walk.
    """
    if isinstance(distribution, Counter):
        return sample_from_counter(distribution)
    probs = list(distribution)
    if values is None:
        raise ValueError("sample() needs 'values' when 'distribution' is a list")
    values = list(values)
    total = float(sum(probs))
    if total == 0:
        # Degenerate distribution -- fall back to a uniform choice.
        return random.choice(values)
    target = random.random() * total
    cumulative = 0.0
    for value, prob in zip(values, probs):
        cumulative += prob
        if cumulative >= target:
            return value
    return values[-1]


def sample_from_counter(ctr):
    """Sample a key from a Counter weighted by its values."""
    items = list(ctr.items())
    return sample([p for _, p in items], [k for k, _ in items])


def raise_not_defined():
    """Raise NotImplementedError, reporting the caller's file/line/method."""
    frame = inspect.stack()[1]
    file_name, line, method = frame[1], frame[2], frame[3]
    message = f"*** Method not implemented: {method} at line {line} of {file_name}"
    print(message, file=sys.stderr)
    raise NotImplementedError(message)
