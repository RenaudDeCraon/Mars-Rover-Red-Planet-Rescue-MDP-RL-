"""
tests/test_grid_layouts.py
--------------------------
Ad-hoc verification script for the Mars Rover grid layouts.

Runs five sanity checks:

1. Print the BaseCampGrid: all states, start state, terminal rewards.
2. Verify transition probabilities sum to 1.0 for every (state, action)
   pair on *every* registered grid.
3. Verify the CanyonGrid layout matches the intended structure:
   start at (0, 2), crater cliff at y=0, close exit +1 at (4, 3),
   far exit +10 at (4, 2).
4. Verify walls properly block movement — the rover stays in place
   (and its stay-in-place mass accumulates via util.Counter when
   multiple slip outcomes collapse there).
5. Verify that the 'extract' action at a reward cell deterministically
   transitions to TERMINAL_STATE with probability 1.0.

Usage::

    python test_grid_layouts.py
"""

import math
import sys

from env.mars_grid import (
    build_mars_grid,
    GRID_REGISTRY,
    TERMINAL_STATE,
    NORTH, SOUTH, EAST, WEST, EXTRACT,
)


# ----------------------------------------------------------------------
# Tiny assertion framework so one failing check doesn't hide the rest
# ----------------------------------------------------------------------


_passed = []
_failed = []


def check(name, condition, detail=''):
    if condition:
        _passed.append(name)
        print('  PASS  {0}'.format(name))
    else:
        _failed.append((name, detail))
        print('  FAIL  {0}{1}'.format(
            name, (' -- ' + detail) if detail else ''
        ))


def section(title):
    print()
    print('=' * 68)
    print(title)
    print('-' * 68)


# ======================================================================
# Test 1: BaseCampGrid inventory
# ======================================================================


def test_base_camp_inventory():
    section('Test 1: BaseCampGrid inventory')

    mdp = build_mars_grid('base_camp', noise=0.2, discount=0.9, living_reward=0.0)

    print('  Grid dimensions : {0} wide x {1} tall'.format(mdp.width, mdp.height))
    print('  Start state     : {0}'.format(mdp.get_start_state()))
    print()
    print('  All states (including absorbing TERMINAL_STATE):')
    states = mdp.get_states()
    for state in states:
        if state == TERMINAL_STATE:
            print('    {0!s:<14}  (absorbing)'.format(state))
            continue
        x, y = state
        cell = mdp.cells[x][y]
        if isinstance(cell, float):
            kind = 'terminal reward = {0:+.1f}'.format(cell)
        else:
            kind = 'open terrain'
        print('    {0!s:<14}  {1}'.format(state, kind))
    print()
    print('  Terminal-reward cells:')
    rewards = [
        (state, mdp.cells[state[0]][state[1]])
        for state in states
        if state != TERMINAL_STATE and isinstance(mdp.cells[state[0]][state[1]], float)
    ]
    for state, reward in rewards:
        label = 'sample site' if reward > 0 else 'crater'
        print('    {0}  =  {1:+.1f}  ({2})'.format(state, reward, label))

    # Minimal structural assertions so the section also contributes to
    # the pass/fail summary at the bottom of the script.
    check(
        'BaseCampGrid has a start state',
        mdp.get_start_state() is not None,
    )
    check(
        'BaseCampGrid has the absorbing TERMINAL_STATE',
        TERMINAL_STATE in states,
    )
    check(
        'BaseCampGrid has at least one positive reward tile',
        any(r > 0 for _, r in rewards),
    )
    check(
        'BaseCampGrid has at least one negative reward tile',
        any(r < 0 for _, r in rewards),
    )


# ======================================================================
# Test 2: transition probabilities sum to 1.0
# ======================================================================


def test_transition_sums():
    section('Test 2: T(s, a, .) probabilities sum to 1.0 on every grid')

    tolerance = 1e-9
    for grid_name in sorted(GRID_REGISTRY):
        mdp = build_mars_grid(grid_name, noise=0.2, discount=0.9, living_reward=0.0)
        bad = []
        checked = 0
        for state in mdp.get_states():
            if mdp.is_terminal(state):
                continue
            for action in mdp.get_possible_actions(state):
                transitions = mdp.get_transition_states_and_probs(state, action)
                total = sum(p for _, p in transitions)
                checked += 1
                if abs(total - 1.0) > tolerance:
                    bad.append((state, action, total))
                # Any negative probabilities are automatically wrong.
                for _, p in transitions:
                    if p < 0:
                        bad.append((state, action, p))
        if bad:
            check(
                '{0}: probability-sum invariant'.format(grid_name),
                False,
                '{0} bad entries (first: {1!r})'.format(len(bad), bad[0]),
            )
        else:
            check(
                '{0}: probability-sum invariant ({1} (s,a) pairs)'.format(
                    grid_name, checked,
                ),
                True,
            )


# ======================================================================
# Test 3: CanyonGrid layout
# ======================================================================


def test_canyon_layout():
    section('Test 3: CanyonGrid layout matches the intended structure')

    mdp = build_mars_grid('canyon', noise=0.2, discount=0.9, living_reward=0.0)

    print('  width = {0}, height = {1}'.format(mdp.width, mdp.height))
    print('  start = {0}'.format(mdp.get_start_state()))
    print()
    print('  cells[x][y] (y=0 is bottom):')
    for y in range(mdp.height - 1, -1, -1):
        row = []
        for x in range(mdp.width):
            cell = mdp.cells[x][y]
            if isinstance(cell, float):
                row.append('{0:+5.1f}'.format(cell))
            elif cell == '#':
                row.append('  #  ')
            else:
                row.append('  .  ')
        print('    y={0}: {1}'.format(y, ' '.join(row)))

    check(
        'start state == (0, 2)',
        mdp.get_start_state() == (0, 2),
        'got {0}'.format(mdp.get_start_state()),
    )
    check(
        'close exit +1 at (4, 3)',
        mdp.cells[4][3] == 1.0,
        'cells[4][3] = {0!r}'.format(mdp.cells[4][3]),
    )
    check(
        'far exit +10 at (4, 2)',
        mdp.cells[4][2] == 10.0,
        'cells[4][2] = {0!r}'.format(mdp.cells[4][2]),
    )
    # Crater cliff along y=0 (every column but the leftmost, which is
    # a wall in the canyon layout).
    cliff_ok = all(mdp.cells[x][0] == -10.0 for x in range(1, mdp.width))
    check(
        'crater cliff (-10) fills y=0 columns 1..{0}'.format(mdp.width - 1),
        cliff_ok,
        'row-0 contents: {0}'.format([mdp.cells[x][0] for x in range(mdp.width)]),
    )


# ======================================================================
# Test 4: wall blocking
# ======================================================================


def test_walls_block_movement():
    section('Test 4: walls and boundaries block movement (stay in place)')

    # base_camp has a wall at (1, 1). From (0, 1) going east the rover
    # should bounce off the wall; from (2, 1) going west likewise;
    # from (1, 0) going north into the wall likewise.
    mdp = build_mars_grid('base_camp', noise=0.0, discount=0.9)
    env = None  # we only test the MDP model here

    # Case A: (0, 1) EAST into the wall at (1, 1)
    trans = dict(mdp.get_transition_states_and_probs((0, 1), EAST))
    check(
        '(0,1) EAST into wall -> stays at (0,1)',
        trans == {(0, 1): 1.0},
        'got {0}'.format(trans),
    )

    # Case B: (2, 1) WEST into the wall at (1, 1)
    trans = dict(mdp.get_transition_states_and_probs((2, 1), WEST))
    check(
        '(2,1) WEST into wall -> stays at (2,1)',
        trans == {(2, 1): 1.0},
        'got {0}'.format(trans),
    )

    # Case C: (1, 0) NORTH into the wall at (1, 1)
    trans = dict(mdp.get_transition_states_and_probs((1, 0), NORTH))
    check(
        '(1,0) NORTH into wall -> stays at (1,0)',
        trans == {(1, 0): 1.0},
        'got {0}'.format(trans),
    )

    # Case D: boundary bounce. (0, 0) WEST -> stays (out of bounds).
    trans = dict(mdp.get_transition_states_and_probs((0, 0), WEST))
    check(
        '(0,0) WEST out-of-bounds -> stays at (0,0)',
        trans == {(0, 0): 1.0},
        'got {0}'.format(trans),
    )

    # Case E: with noise > 0, wall + slip outcomes collapse to the
    # same cell and the probabilities accumulate via util.Counter.
    # From (1, 0) going NORTH with noise=0.2:
    #   intended north: wall at (1,1) -> stay (1,0), p=0.8
    #   left slip west: (0,0), p=0.1
    #   right slip east: (2,0), p=0.1
    noisy = build_mars_grid('base_camp', noise=0.2, discount=0.9)
    trans = dict(noisy.get_transition_states_and_probs((1, 0), NORTH))
    expected = {(1, 0): 0.8, (0, 0): 0.1, (2, 0): 0.1}
    ok = (set(trans.keys()) == set(expected.keys())
          and all(abs(trans[k] - expected[k]) < 1e-9 for k in expected))
    check(
        'noisy (1,0) NORTH stay-in-place + slips accumulate correctly',
        ok,
        'got {0}, expected {1}'.format(trans, expected),
    )


# ======================================================================
# Test 5: extract -> TERMINAL_STATE
# ======================================================================


def test_extract_terminates():
    section("Test 5: 'extract' at a reward cell -> TERMINAL_STATE w.p. 1.0")

    # Exercise both a positive and a negative terminal.
    mdp = build_mars_grid('base_camp', noise=0.2, discount=0.9)

    # Locate a positive and a negative terminal cell.
    positive_cells = []
    negative_cells = []
    for state in mdp.get_states():
        if state == TERMINAL_STATE:
            continue
        x, y = state
        cell = mdp.cells[x][y]
        if isinstance(cell, float) and cell > 0:
            positive_cells.append((state, cell))
        elif isinstance(cell, float) and cell < 0:
            negative_cells.append((state, cell))

    # Make sure we have something to test on.
    check(
        'base_camp has at least one positive reward tile',
        len(positive_cells) >= 1,
    )
    check(
        'base_camp has at least one negative reward tile',
        len(negative_cells) >= 1,
    )

    for state, reward in positive_cells + negative_cells:
        actions = mdp.get_possible_actions(state)
        check(
            "{0}: only 'extract' action available".format(state),
            actions == (EXTRACT,),
            'got {0}'.format(actions),
        )
        trans = mdp.get_transition_states_and_probs(state, EXTRACT)
        check(
            '{0} EXTRACT -> TERMINAL_STATE w.p. 1.0'.format(state),
            trans == [(TERMINAL_STATE, 1.0)],
            'got {0}'.format(trans),
        )
        paid = mdp.get_reward(state, EXTRACT, TERMINAL_STATE)
        check(
            '{0} EXTRACT reward equals {1:+.1f}'.format(state, reward),
            paid == reward,
            'got {0}'.format(paid),
        )


# ======================================================================
# Main
# ======================================================================


def main():
    test_base_camp_inventory()
    test_transition_sums()
    test_canyon_layout()
    test_walls_block_movement()
    test_extract_terminates()

    print()
    print('=' * 68)
    print('Summary: {0} passed, {1} failed'.format(len(_passed), len(_failed)))
    if _failed:
        print('Failures:')
        for name, detail in _failed:
            print('  * {0}{1}'.format(name, (' -- ' + detail) if detail else ''))
        sys.exit(1)
    print('All grid-layout checks passed.')


if __name__ == '__main__':
    main()
