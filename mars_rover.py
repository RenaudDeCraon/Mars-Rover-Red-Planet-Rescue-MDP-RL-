"""
mars_rover.py -- main entry point for Mars Rover: Red Planet Rescue.

CS 451/551 (Introduction to AI) at Ozyegin University.

Usage::

    python mars_rover.py                              # random agent, base_camp
    python mars_rover.py -m                            # manual keyboard control
    python mars_rover.py -a value -i 100 -k 10         # value iteration
    python mars_rover.py -a q -k 100                  # Q-learning
    python mars_rover.py -g canyon_grid                # specific grid
    python mars_rover.py -t                            # text mode (no GUI)
    python mars_rover.py --list-layouts                # show available grids
    python mars_rover.py --discount 0.9 --noise 0.2 --living-reward 0.0
"""

import argparse
import os
import random
import re
import sys
import time
from typing import Any

import util
from env.mars_grid import (
    MarsGridEnvironment, GRID_REGISTRY, build_mars_grid,
    TERMINAL_STATE, get_layout_path,
)

BANNER = r"""
    ======================================================
       __  __                   ____
      |  \/  |   __ _   _ __   / ___|     .-"\"-.__
      | |\/| |  / _` | | '__|  \___ \    /       \\
      | |  | | | (_| | | |      ___) |  |  o   o  |
      |_|  |_|  \__,_| |_|     |____/    \_ .o. _/
                                            '---'
        Mars Rover: Red Planet Rescue
        CS 451/551 -- Intro to AI -- Rover-7 Mission
    ======================================================
"""


# ------------------------------------------------------------------
# Grid name resolution
# ------------------------------------------------------------------

def _available_layouts() -> list[str]:
    """Return every layout name available (.lay files + registry)."""
    names: set[str] = set(GRID_REGISTRY.keys())
    lay_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layouts')
    if os.path.isdir(lay_dir):
        for f in os.listdir(lay_dir):
            if f.endswith('.lay'):
                names.add(f[:-4])
    return sorted(names)


def _resolve_grid(name: str) -> str:
    """Normalise user-typed grid name (accepts CamelCase + Grid suffix)."""
    # Exact match in registry or .lay file?
    if name in GRID_REGISTRY or os.path.exists(get_layout_path(name)):
        return name
    stripped = name[:-4] if name.endswith('Grid') else name
    snake = re.sub(r'(?<!^)(?=[A-Z])', '_', stripped).lower()
    if snake in GRID_REGISTRY or os.path.exists(get_layout_path(snake)):
        return snake
    if name.lower() in GRID_REGISTRY:
        return name.lower()
    raise ValueError(f"Unknown grid {name!r}. Available: {_available_layouts()}")


# ------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------

class RandomAgent:
    """Returns None from get_action; the episode loop picks at random."""
    def get_action(self, state: Any) -> None:
        return None


def _build_agent(args, env, mdp):
    name = args.agent.lower()
    if name == 'random':
        return RandomAgent()
    if name == 'value':
        from students.value_iteration_agents import ValueIterationAgent
        return ValueIterationAgent(mdp, discount=args.discount,
                                   iterations=args.iterations)
    if name == 'q':
        from students.qlearning_agents import QLearningAgent
        return QLearningAgent(action_fn=env.get_possible_actions,
                              num_training=max(args.episodes, 1),
                              epsilon=args.epsilon, alpha=args.alpha,
                              gamma=args.discount)
    raise ValueError(f"Unknown agent {name!r}. Use 'random', 'value', or 'q'.")


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

def _load_display(mdp, text_mode: bool):
    """Try the GUI display, fall back to text, return (display, is_gui)."""
    if not text_mode:
        try:
            from display.gui_display import MarsDisplay
            return MarsDisplay(mdp), True
        except Exception:
            pass
    try:
        from display.text_display import TextMarsDisplay
        return TextMarsDisplay(mdp), False
    except Exception:
        return None, False


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

def run_episode(env, agent, *, discount=0.9, max_steps=200,
                display=None, delay=0.0, quiet=False, manual=False):
    env.reset()
    if hasattr(agent, 'start_episode'):
        agent.start_episode()

    total_r = disc_r = 0.0
    gamma_pow = 1.0
    steps = 0

    while not env.is_terminal() and steps < max_steps:
        state = env.get_current_state()
        legal = env.get_possible_actions(state)
        if not legal:
            break

        if manual:
            if display is not None:
                try:
                    display.draw(env=env, agent=agent, state=state, step=steps)
                except Exception:
                    pass
            action = _manual_action(state, legal, display)
        else:
            action = agent.get_action(state)
            if action is None:
                action = random.choice(legal)

        next_state, reward = env.do_action(action)
        steps += 1
        if hasattr(agent, 'observe_transition'):
            agent.observe_transition(state, action, next_state, reward)

        total_r += reward
        disc_r += gamma_pow * reward
        gamma_pow *= discount

        if not quiet:
            print(f"    step {steps:3d}: {state} --{action}--> {next_state}  r={reward:+.3f}")

        if display is not None:
            try:
                display.draw(env=env, agent=agent, state=next_state,
                             action=action, reward=reward, step=steps)
            except Exception:
                pass
            if hasattr(display, 'should_quit') and display.should_quit:
                break
        if delay > 0:
            time.sleep(delay)

    if hasattr(agent, 'stop_episode'):
        agent.stop_episode()

    return {'return': total_r, 'discounted_return': disc_r,
            'steps': steps, 'terminated': env.is_terminal()}


def _manual_action(state, legal, display):
    if display and hasattr(display, 'wait_for_key_action'):
        return display.wait_for_key_action(legal)
    print(f"\n  State: {state}  Legal: {list(legal)}")
    print("  > ", end='', flush=True)
    try:
        line = sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return random.choice(legal)
    if not line:
        return random.choice(legal)
    if line in legal:
        return line
    matches = [a for a in legal if a.startswith(line)]
    return matches[0] if len(matches) == 1 else random.choice(legal)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(prog='mars_rover.py',
                                description='Mars Rover: Red Planet Rescue')
    p.add_argument('-g', '--grid', default='base_camp')
    p.add_argument('-a', '--agent', default='random',
                   choices=['random', 'value', 'q'])
    p.add_argument('-m', '--manual', action='store_true')
    p.add_argument('-t', '--text', action='store_true')
    p.add_argument('-q', '--quiet', action='store_true')
    p.add_argument('--discount', type=float, default=0.9)
    p.add_argument('--noise', type=float, default=0.2)
    p.add_argument('--living-reward', type=float, default=0.0)
    p.add_argument('-i', '--iterations', type=int, default=100)
    p.add_argument('-k', '--episodes', type=int, default=10)
    p.add_argument('-e', '--epsilon', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--speed', type=float, default=0.0)
    p.add_argument('-s', '--seed', type=int, default=None)
    p.add_argument('--list-layouts', action='store_true',
                   help='list available grid layouts and exit')
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.list_layouts:
        print('Available layouts:')
        for name in _available_layouts():
            print(f'  {name}')
        return

    print(BANNER)
    if args.seed is not None:
        random.seed(args.seed)

    grid_name = _resolve_grid(args.grid)
    mdp = build_mars_grid(grid_name, noise=args.noise,
                          living_reward=args.living_reward,
                          discount=args.discount)
    env = MarsGridEnvironment(mdp)

    print(f"Grid:          {grid_name}")
    print(f"Agent:         {args.agent}")
    print(f"Episodes:      {args.episodes}")
    print(f"Discount:      {args.discount}")
    print(f"Noise:         {args.noise}")
    print(f"Living reward: {args.living_reward}")
    if args.agent == 'value':
        print(f"VI iterations: {args.iterations}")
    if args.agent == 'q':
        print(f"Epsilon:       {args.epsilon}")
        print(f"Alpha:         {args.alpha}")
    if args.seed is not None:
        print(f"Random seed:   {args.seed}")
    print()

    agent = _build_agent(args, env, mdp)
    display, is_gui = _load_display(mdp, args.text)
    if display is None:
        print("(no display available)")
    elif not is_gui:
        print("(text display)")
    print()

    returns: list[float] = []
    for ep in range(1, args.episodes + 1):
        if not args.quiet:
            print(f"--- Episode {ep}/{args.episodes} ---")
        result = run_episode(env, agent, discount=args.discount,
                             display=display, delay=args.speed,
                             quiet=args.quiet, manual=args.manual)
        returns.append(result['discounted_return'])
        if not args.quiet:
            print(f"  return={result['return']:+.3f}  "
                  f"discounted={result['discounted_return']:+.3f}  "
                  f"steps={result['steps']}  term={result['terminated']}")
        if is_gui and hasattr(display, 'should_quit') and display.should_quit:
            break

    print()
    if returns:
        avg = sum(returns) / len(returns)
        print(f"Average discounted return over {len(returns)} episodes: {avg:.4f}")

    if display is not None and hasattr(display, 'close'):
        try:
            display.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
