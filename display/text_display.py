"""
display/text_display.py -- ASCII text display for the Mars grid (3.10+).

Part of the "Mars Rover: Red Planet Rescue" homework for CS 451/551.

Renders a :class:`env.mars_grid.MarsGrid` to the terminal as an
ASCII grid bounded by Unicode box-drawing characters (``┌ ─ ┬ ┐``
etc.), with each cell showing one of:

* ``######`` for walls,
* the reward value for terminal reward cells (``+1``, ``-10``, ...),
* the rover marker ``R`` for the rover's current position, or
* the agent's ``V(s)`` value or ``pi(s)`` policy arrow for every
  other open cell, depending on the chosen *display mode*.

The class is a simple duck-typed match for the interface the
training loop expects: a :meth:`draw` method that accepts
arbitrary ``**kwargs``, an :meth:`handle_events` no-op for input
processing, :meth:`wait_for_key` for "press enter to continue"
prompts, and :meth:`close` for cleanup. It has no dependencies
on Pygame, so it runs on headless machines and CI boxes.

Typical use::

    from env.mars_grid import build_mars_grid
    from display.text_display import TextMarsDisplay

    mdp = build_mars_grid('base_camp')
    display = TextMarsDisplay(mdp, mode='values')
    display.draw(env=env, agent=agent)
"""

import sys
from typing import Any, Iterator


class TextMarsDisplay:
    """Pure-text Mars grid renderer.

    Parameters
    ----------
    mdp : env.mars_grid.MarsGrid or equivalent
        Must expose ``width``, ``height``, and ``cells[x][y]``. Both
        :class:`env.mars_grid.MarsGrid` and the flat-layout
        ``mars_grid.MarsGrid`` satisfy this.
    mode : str
        Default display mode, one of:

        * ``'values'``  -- show ``V(state)`` inside each open cell
          (requires an ``agent`` in :meth:`draw`'s kwargs).
        * ``'policy'``  -- show the policy arrow at each open cell.
        * ``'grid'``    -- show just the terrain, no agent info.

        Individual :meth:`draw` calls can override the default via
        ``mode=...`` in kwargs.
    """

    # Every cell is ``_CELL_WIDTH`` characters wide. Six is a good
    # compromise: it fits "+100.5" and "-100.5" exactly, which are
    # the worst cases for value iteration on the shipped grids.
    _CELL_WIDTH: int = 6

    # Unicode arrows for policy rendering. Single-codepoint chars
    # so they occupy one column in most terminals.
    _ARROWS: dict[str, str] = {
        'north': '↑',
        'south': '↓',
        'east': '→',
        'west': '←',
        'extract': '*',
    }

    # Fallback marker for an action the display does not recognise.
    _UNKNOWN_ARROW: str = '?'

    def __init__(self, mdp: Any, mode: str = 'values'):
        self.mdp = mdp
        self.mode = mode

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def draw(self, **kwargs: Any) -> None:
        """Print one rendering of the grid to stdout.

        Recognised kwargs (all optional):

        * ``env``    -- an environment with ``get_current_state()``
          used to place the rover marker. Overridden by ``state``.
        * ``state``  -- an explicit rover coordinate, overriding
          ``env.get_current_state()``.
        * ``agent``  -- an agent exposing ``get_value`` /
          ``get_policy``; required for the 'values' and 'policy'
          modes, ignored in 'grid' mode.
        * ``mode``   -- per-call display mode override.
        * ``header`` -- optional string printed above the grid.
        * ``footer`` -- optional string printed below the grid.
        * ``title``  -- alias for ``header``.
        * ``step`` / ``action`` / ``reward`` / ``next_state`` --
          training-loop metadata; if ``header`` is absent, they
          are collapsed into a one-line HUD above the grid.

        Any other kwargs are silently ignored so the training loop
        can pass whatever it has without the display crashing.
        """
        mode = kwargs.get('mode', self.mode)

        # Where is the rover?
        state = kwargs.get('state')
        if state is None:
            env = kwargs.get('env')
            if env is not None and hasattr(env, 'get_current_state'):
                state = env.get_current_state()

        agent = kwargs.get('agent')

        # Optional header. Explicit 'header' / 'title' wins; else
        # build a one-line HUD from any training metadata present.
        header = kwargs.get('header') or kwargs.get('title')
        if header is None and any(k in kwargs for k in ('step', 'action', 'reward')):
            header = self._format_hud(kwargs)

        if header is not None:
            print(header)

        for line in self._render_lines(state, agent, mode):
            print(line)

        footer = kwargs.get('footer')
        if footer is not None:
            print(footer)

    def handle_events(self) -> None:
        """Process pending input events -- no-op for the text display."""
        return None

    def wait_for_key(self) -> None:
        """Block until the user presses Enter (or EOF on stdin)."""
        try:
            print('[press Enter to continue]', end='', flush=True)
            sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            pass

    def close(self) -> None:
        """No resources to release in text mode."""
        return None

    # ------------------------------------------------------------------
    # Rendering internals
    # ------------------------------------------------------------------

    def _render_lines(
        self,
        rover_state: Any,
        agent: Any,
        mode: str,
    ) -> Iterator[str]:
        """Yield the box-drawn lines of the grid, top row first."""
        width = self.mdp.width
        height = self.mdp.height
        cell_w = self._CELL_WIDTH
        horizontal = '─' * cell_w

        # Pre-compute the three kinds of horizontal separator.
        top    = '┌' + '┬'.join([horizontal] * width) + '┐'
        middle = '├' + '┼'.join([horizontal] * width) + '┤'
        bottom = '└' + '┴'.join([horizontal] * width) + '┘'

        yield top
        # y grows upward in the MDP; the top of the printed grid is
        # the highest y.
        for y in range(height - 1, -1, -1):
            cells = [
                self._cell_content((x, y), rover_state, agent, mode)
                for x in range(width)
            ]
            yield '│' + '│'.join(cells) + '│'
            if y > 0:
                yield middle
        yield bottom

    def _cell_content(
        self,
        state: tuple[int, int],
        rover_state: Any,
        agent: Any,
        mode: str,
    ) -> str:
        """Return the ``_CELL_WIDTH``-character body for one cell.

        Content priority: wall > rover > reward-terminal > (value
        or policy depending on mode) > blank.
        """
        x, y = state
        cell = self.mdp.cells[x][y]

        # 1. Walls -- always a solid block.
        if cell == '#':
            return '#' * self._CELL_WIDTH

        # 2. Rover marker takes precedence over everything else.
        if rover_state is not None and state == rover_state:
            return self._pad('R')

        # 3. Terminal reward cell.
        if isinstance(cell, (int, float)) and not isinstance(cell, bool):
            return self._format_reward(float(cell))

        # 4. Open cell: show value / policy / blank based on mode.
        if agent is None or mode in ('grid', 'none'):
            return ' ' * self._CELL_WIDTH

        if mode == 'values':
            try:
                value = agent.get_value(state)
            except Exception:
                return ' ' * self._CELL_WIDTH
            return self._format_value(float(value))

        if mode == 'policy':
            try:
                action = agent.get_policy(state)
            except Exception:
                return ' ' * self._CELL_WIDTH
            return self._format_action(action)

        return ' ' * self._CELL_WIDTH

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _pad(self, text: str) -> str:
        """Center ``text`` in a cell-wide field, truncating if needed."""
        return text.center(self._CELL_WIDTH)[: self._CELL_WIDTH]

    def _format_reward(self, value: float) -> str:
        """Format a terminal reward to fit in one cell."""
        if value == int(value):
            s = f'{int(value):+d}'
        else:
            s = f'{value:+.1f}'
        return self._pad(s)

    def _format_value(self, value: float) -> str:
        """Format a V-value to fit in one cell.

        Width-adaptive precision: 2 decimals when ``|value| < 10``,
        1 decimal up to 100, no decimals for larger magnitudes.
        Value 0.0 renders as a lone dot so untouched cells are
        visually distinguishable from converged zero-value cells.
        """
        if value == 0:
            return self._pad('.')
        abs_v = abs(value)
        if abs_v < 10:
            s = f'{value:+.2f}'    # "+0.66"
        elif abs_v < 100:
            s = f'{value:+.1f}'    # "+12.3"
        else:
            s = f'{value:+.0f}'    # "+123"
        return self._pad(s)

    def _format_action(self, action: Any) -> str:
        """Format a policy action as a centred arrow."""
        if action is None:
            return self._pad('.')
        arrow = self._ARROWS.get(action, self._UNKNOWN_ARROW)
        return self._pad(arrow)

    def _format_hud(self, kwargs: dict) -> str:
        """Collapse training-loop metadata into a one-line HUD."""
        parts: list[str] = []
        if 'step' in kwargs:
            parts.append(f'step={kwargs["step"]}')
        if 'action' in kwargs:
            parts.append(f'action={kwargs["action"]}')
        if 'reward' in kwargs:
            parts.append(f'reward={kwargs["reward"]:+.3f}')
        if 'next_state' in kwargs:
            parts.append(f'next={kwargs["next_state"]}')
        return '  '.join(parts)
