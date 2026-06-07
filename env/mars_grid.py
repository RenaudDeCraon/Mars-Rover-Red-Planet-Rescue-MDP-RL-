"""
env/mars_grid.py -- the Mars Rover grid-world MDP and its wrappers.

Part of the "Mars Rover: Red Planet Rescue" homework for CS 451/551.

Defines:

* :class:`MarsGrid` -- a stochastic 2-D grid-world MDP. Open cells are
  safe Martian terrain, ``'#'`` cells are impassable boulders, and
  numeric cells are *sample-extraction* sites whose reward is paid
  out when the rover performs the dedicated ``'extract'`` action
  (positive values are geological samples or base camps; negative
  values are craters). Movement is corrupted by 20% solar-interference
  noise by default: with probability ``1 - noise`` the rover moves in
  its intended direction, and with ``noise / 2`` each it slips to a
  perpendicular. Walls and grid boundaries bounce the rover in place,
  and multiple outcomes that collapse to the same square accumulate
  via :class:`util.Counter`.
* :class:`MarsGridEnvironment` -- an interactive wrapper around a
  :class:`MarsGrid` so that model-free learners can experience the
  world one sample at a time without ever calling the MDP's
  transition function directly.
* :func:`load_grid_from_file` -- parses a ``.lay`` layout file into a
  ``grid_text`` list-of-lists suitable for the :class:`MarsGrid`
  constructor.
* :func:`get_layout_path` -- locates ``layouts/<name>.lay`` relative
  to the project root.
* :func:`build_mars_grid` -- dispatch entry point that tries to load
  ``layouts/<name>.lay`` first and falls back to the hardcoded
  :data:`GRID_REGISTRY` if the file is not there.
* :data:`GRID_REGISTRY` -- dict of grid name -> callable returning
  ``grid_text``, serving as both the fallback for missing layout
  files and a source of readily-constructible test worlds.
"""

import os
from typing import Any

import util
from env.mdp import MarkovDecisionProcess
from env.environment import Environment


# The single absorbing terminal state. Using a string sentinel (rather
# than ``None``) makes the state show up clearly in debug output.
TERMINAL_STATE = 'TERMINAL_STATE'

# Action name constants, exported via the module namespace so other
# modules can import them symbolically instead of repeating strings.
NORTH = 'north'
SOUTH = 'south'
EAST = 'east'
WEST = 'west'
EXTRACT = 'extract'

# Unit vector per direction. y grows upward, so 'north' is +y.
_DELTAS = {
    NORTH: (0, 1),
    SOUTH: (0, -1),
    EAST: (1, 0),
    WEST: (-1, 0),
}

# "Left of north is west", etc. -- used to compute slip directions.
_LEFT_OF = {NORTH: WEST, WEST: SOUTH, SOUTH: EAST, EAST: NORTH}
_RIGHT_OF = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}


class MarsGrid(MarkovDecisionProcess):
    """A fully-observable stochastic grid world for Rover-7."""

    def __init__(
        self,
        grid_text: list[list],
        noise: float = 0.2,
        living_reward: float = 0.0,
        discount: float = 0.9,
    ):
        self.noise = noise
        self.living_reward = living_reward
        self.discount = discount

        # Dimensions are inferred from the input. Non-rectangular
        # input will raise an IndexError, which is an acceptable
        # fail-fast for a malformed world.
        self.height = len(grid_text)
        self.width = len(grid_text[0])

        # Store cells indexed as ``self.cells[x][y]`` with y=0 at the
        # bottom row. Flipping at parse time keeps every downstream
        # coordinate calculation in world-space.
        self.cells: list[list] = [[None] * self.height for _ in range(self.width)]
        start_state = None
        for row_index, row in enumerate(grid_text):
            y = self.height - 1 - row_index
            for x, raw_cell in enumerate(row):
                parsed = self._parse_cell(raw_cell)
                if parsed == 'S':
                    if start_state is not None:
                        raise ValueError('Multiple start states defined in grid')
                    start_state = (x, y)
                    parsed = ' '
                self.cells[x][y] = parsed

        # Default start = (0, 0) if no 'S' marker was given.
        self.start_state = start_state if start_state is not None else (0, 0)

    # ------------------------------------------------------------------
    # Cell parsing and classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cell(raw: Any) -> Any:
        """Normalise a raw grid entry to ' ', '#', 'S', or a float."""
        if isinstance(raw, bool):
            # Booleans subclass int in Python; rule them out explicitly
            # so "True" doesn't silently become a +1 terminal.
            raise ValueError('Grid cells may not be booleans')
        if isinstance(raw, (int, float)):
            return float(raw)
        if raw is None:
            return ' '
        if not isinstance(raw, str):
            raise ValueError(
                f'Grid cells must be str, int, float, or None; got {type(raw).__name__}'
            )
        stripped = raw.strip()
        if stripped == '':
            return ' '
        if stripped not in ('#', 'S'):
            raise ValueError(f'Unrecognised grid cell: {raw!r}')
        return stripped

    def _cell_at(self, state: tuple[int, int]) -> Any:
        x, y = state
        return self.cells[x][y]

    @staticmethod
    def _is_reward_cell(cell: Any) -> bool:
        return isinstance(cell, float) and not isinstance(cell, bool)

    # ------------------------------------------------------------------
    # MarkovDecisionProcess interface
    # ------------------------------------------------------------------

    def get_states(self) -> list[Any]:
        """Return every reachable state, plus the absorbing terminal."""
        states: list[Any] = [TERMINAL_STATE]
        for x in range(self.width):
            for y in range(self.height):
                if self.cells[x][y] != '#':
                    states.append((x, y))
        return states

    def get_start_state(self) -> tuple[int, int]:
        return self.start_state

    def get_possible_actions(self, state: Any) -> tuple[str, ...]:
        if state == TERMINAL_STATE:
            return ()
        cell = self._cell_at(state)
        if cell == '#':
            return ()
        if self._is_reward_cell(cell):
            return (EXTRACT,)
        return (NORTH, SOUTH, EAST, WEST)

    def get_transition_states_and_probs(
        self, state: Any, action: str
    ) -> list[tuple[Any, float]]:
        if state == TERMINAL_STATE:
            return []
        cell = self._cell_at(state)

        if self._is_reward_cell(cell):
            if action == EXTRACT:
                return [(TERMINAL_STATE, 1.0)]
            return []

        if cell == '#':
            return []

        if action not in _DELTAS:
            return []

        intended = self._attempt_move(state, action)
        left = self._attempt_move(state, _LEFT_OF[action])
        right = self._attempt_move(state, _RIGHT_OF[action])

        # Counter accumulates collisions: e.g. if slip-left and
        # intended both land on the same square because of a wall
        # bounce, their probabilities sum instead of overwriting.
        distribution = util.Counter()
        distribution[intended] += 1.0 - self.noise
        distribution[left] += self.noise / 2.0
        distribution[right] += self.noise / 2.0

        return [(s, p) for s, p in distribution.items() if p > 0]

    def _attempt_move(self, state: tuple[int, int], direction: str) -> tuple[int, int]:
        """One-step deterministic move; returns ``state`` on any collision."""
        dx, dy = _DELTAS[direction]
        nx, ny = state[0] + dx, state[1] + dy
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            return state
        if self.cells[nx][ny] == '#':
            return state
        return (nx, ny)

    def get_reward(self, state: Any, action: Any, next_state: Any) -> float:
        if state == TERMINAL_STATE:
            return 0.0
        cell = self._cell_at(state)
        if self._is_reward_cell(cell):
            if action == EXTRACT:
                return cell
            return 0.0
        return self.living_reward

    def is_terminal(self, state: Any) -> bool:
        """Only the absorbing ``TERMINAL_STATE`` is terminal."""
        return state == TERMINAL_STATE


class MarsGridEnvironment(Environment):
    """Interactive wrapper that lets an RL agent experience a MarsGrid.

    Stores the rover's current position internally and exposes only
    the sampling interface from :class:`Environment`. Model-free
    learners should drive the world through this class; the underlying
    MDP is still accessible via ``env.grid_mdp`` for rendering,
    planning, and grading code.
    """

    def __init__(self, grid_mdp: MarsGrid):
        self.grid_mdp = grid_mdp
        self.state: Any = None
        self.reset()

    def get_current_state(self) -> Any:
        return self.state

    def get_possible_actions(self, state: Any) -> tuple[str, ...]:
        return self.grid_mdp.get_possible_actions(state)

    def do_action(self, action: str) -> tuple[Any, float]:
        """Sample a successor state, update ``self.state``, return ``(s', r)``."""
        state = self.state
        successors = self.grid_mdp.get_transition_states_and_probs(state, action)
        if not successors:
            # Illegal action from this state: stay put, pay whatever
            # the MDP's get_reward decides (usually 0).
            reward = self.grid_mdp.get_reward(state, action, state)
            return (state, reward)

        probs = [prob for _, prob in successors]
        next_states = [next_state for next_state, _ in successors]
        next_state = util.sample(probs, next_states)
        reward = self.grid_mdp.get_reward(state, action, next_state)
        self.state = next_state
        return (next_state, reward)

    def reset(self) -> None:
        self.state = self.grid_mdp.get_start_state()

    def is_terminal(self) -> bool:
        return self.grid_mdp.is_terminal(self.state)


# ----------------------------------------------------------------------
# Layout file I/O
# ----------------------------------------------------------------------
#
# A .lay file is a plain-text grid, one row per non-blank line, tokens
# separated by whitespace. Tokens:
#
#     _ or .      open terrain
#     #           wall
#     S           start position (also open terrain)
#     int/float   terminal reward value
#
# Lines starting with ``//`` are comments (note that ``#`` is reserved
# for walls, so we use C-style ``//`` for comments instead).
# Rows must all be the same length; an empty file is an error.


def load_grid_from_file(filepath: str) -> list[list]:
    """Parse a .lay file into a ``grid_text`` list-of-lists.

    Returns a list of rows, each a list of cell tokens suitable for
    the :class:`MarsGrid` constructor. Raises :class:`ValueError` on
    empty files, non-rectangular layouts, or unrecognised tokens.
    """
    rows: list[list] = []
    with open(filepath, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('//'):
                continue
            row: list = []
            for token in line.split():
                if token in ('_', '.'):
                    row.append(' ')
                elif token == '#':
                    row.append('#')
                elif token == 'S':
                    row.append('S')
                else:
                    # Try int first, then float. A bare 'int' stays
                    # an int and is cast to float by MarsGrid._parse_cell;
                    # a float literal preserves its decimal.
                    try:
                        value: Any = int(token)
                    except ValueError:
                        try:
                            value = float(token)
                        except ValueError as exc:
                            raise ValueError(
                                f'{filepath}: unrecognised token {token!r}'
                            ) from exc
                    row.append(value)
            rows.append(row)

    if not rows:
        raise ValueError(f'empty layout file: {filepath}')

    width = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f'{filepath}: row {i} has {len(row)} cells, expected {width}'
            )
    return rows


def get_layout_path(name: str) -> str:
    """Return the path to ``layouts/<name>.lay`` relative to the project root.

    ``name`` may be supplied with or without the ``.lay`` extension.
    The project root is deduced from the location of this file: we
    sit in ``mars_rover_rl/env/``, so the project root is the parent
    directory and ``layouts/`` is a sibling of ``env/``.
    """
    if not name.endswith('.lay'):
        name = name + '.lay'
    here = os.path.dirname(os.path.abspath(__file__))  # mars_rover_rl/env/
    project_root = os.path.dirname(here)               # mars_rover_rl/
    return os.path.join(project_root, 'layouts', name)


# ----------------------------------------------------------------------
# Hardcoded fallback layouts
# ----------------------------------------------------------------------
#
# GRID_REGISTRY maps a canonical grid name to a no-arg callable that
# returns a fresh ``grid_text`` list-of-lists. These act as the
# fallback for build_mars_grid when the .lay file is missing, and are
# also useful for unit tests that want to construct a world without
# touching the filesystem.


def get_base_camp_grid() -> list[list]:
    """4 x 3 grid analogous to Berkeley's BookGrid."""
    return [
        [' ', ' ', ' ', -1],
        [' ', '#', ' ', ' '],
        [' ', ' ', ' ',  1],
    ]


def get_canyon_grid() -> list[list]:
    """5 x 4 cliffside grid used for the Q2 discounting analysis."""
    return [
        [' ', ' ', ' ', ' ',  1],
        ['S', ' ', ' ', ' ', 10],
        [' ', ' ', ' ', ' ', ' '],
        ['#', -10, -10, -10, -10],
    ]


def get_maze_grid() -> list[list]:
    """4 x 5 twisty maze with a single +1 sample site at the top-right."""
    return [
        [' ', ' ', ' ',  1],
        ['#', '#', ' ', '#'],
        [' ', ' ', ' ', ' '],
        [' ', '#', '#', ' '],
        ['S', ' ', ' ', ' '],
    ]


def get_expedition_grid() -> list[list]:
    """7 x 6 grid with multiple sample sites and a single crater."""
    return [
        [' ', ' ', ' ', ' ', ' ', ' ',  5],
        [' ', '#', '#', '#', '#', ' ', ' '],
        [' ', ' ', ' ',  2, ' ', ' ', ' '],
        [' ', '#', ' ', '#', '#', '#', ' '],
        [' ', ' ', ' ', ' ', ' ', -5, ' '],
        ['S', ' ', '#', ' ', ' ', ' ',  3],
    ]


def get_small_mars_grid() -> list[list]:
    """3 x 2 micro-grid used by the Q-learning unit tests."""
    return [
        ['#', ' ',  1],
        ['S', ' ', -1],
    ]


def get_medium_mars_grid() -> list[list]:
    """5 x 5 grid with corridors and distributed hazards."""
    return [
        ['S', ' ', ' ', ' ', ' '],
        [' ', '#', ' ', '#', ' '],
        [' ', ' ', ' ', ' ',  1],
        [' ', '#', ' ', '#', ' '],
        [' ', ' ', ' ', ' ', -1],
    ]


GRID_REGISTRY = {
    'base_camp':   get_base_camp_grid,
    'canyon':      get_canyon_grid,
    'maze':        get_maze_grid,
    'expedition':  get_expedition_grid,
    'small_mars':  get_small_mars_grid,
    'medium_mars': get_medium_mars_grid,
}


# ----------------------------------------------------------------------
# Dispatch entry point
# ----------------------------------------------------------------------


def build_mars_grid(
    name: str,
    noise: float = 0.2,
    living_reward: float = 0.0,
    discount: float = 0.9,
) -> MarsGrid:
    """Construct a :class:`MarsGrid` for the named layout.

    Tries ``layouts/<name>.lay`` first; if that file is missing,
    falls back to :data:`GRID_REGISTRY`. Raises :class:`ValueError`
    with a helpful message if the name isn't findable either way.
    """
    path = get_layout_path(name)
    if os.path.exists(path):
        grid_text = load_grid_from_file(path)
    elif name in GRID_REGISTRY:
        grid_text = GRID_REGISTRY[name]()
    else:
        raise ValueError(
            f"Unknown grid {name!r}. Tried layout file {path!r} "
            f"and fallback registry keys {sorted(GRID_REGISTRY)}"
        )
    return MarsGrid(
        grid_text,
        noise=noise,
        living_reward=living_reward,
        discount=discount,
    )
