"""check.py -- quick student self-check (no grade, no points).
Smoke-tests on small_grid/base_camp/canyon. Run autograder.py for actual grade.
"""
import argparse, io, random, sys
import util
from env.mars_grid import build_mars_grid, MarsGridEnvironment, TERMINAL_STATE

_p = _f = 0

def check(label, cond, detail=''):
    global _p, _f
    if cond: _p += 1; print(f'  \u2705 {label}')
    else: _f += 1; print(f'  \u274C {label}' + (f' -- {detail}' if detail else ''))

def _quiet(fn, *a, **kw):
    old = sys.stdout; sys.stdout = io.StringIO()
    try: return fn(*a, **kw)
    finally: sys.stdout = old

def _vi(grid, disc=0.9, noise=0.2, lr=0.0, it=100):
    mdp = build_mars_grid(grid, noise=noise, living_reward=lr, discount=disc)
    from students.value_iteration_agents import ValueIterationAgent
    return mdp, ValueIterationAgent(mdp, discount=disc, iterations=it)

def _ql(grid, disc=0.9, noise=0.2, lr=0.0, eps=0.3, alpha=0.5, n=100):
    mdp = build_mars_grid(grid, noise=noise, living_reward=lr, discount=disc)
    env = MarsGridEnvironment(mdp)
    from students.qlearning_agents import QLearningAgent
    return mdp, env, QLearningAgent(action_fn=env.get_possible_actions,
                                     num_training=n, epsilon=eps, alpha=alpha, gamma=disc)

def _train(agent, env, n):
    for _ in range(n):
        env.reset(); agent.start_episode()
        for _ in range(100):
            if env.is_terminal(): break
            s = env.get_current_state(); a = agent.get_action(s)
            if a is None: break
            ns, r = env.do_action(a); agent.observe_transition(s, a, ns, r)
        agent.stop_episode()

def q1():
    print('\nQ1: Value Iteration')
    mdp, ag = _vi('base_camp')
    start = mdp.get_start_state()
    check('V(start) non-zero', ag.get_value(start) != 0)
    ok = all(
        abs(ag.get_value(s) - max(ag.get_q_value(s, a)
            for a in mdp.get_possible_actions(s))) < 0.001
        for s in mdp.get_states()
        if s != TERMINAL_STATE and mdp.get_possible_actions(s))
    check('V(s) == max_a Q(s,a)', ok)
    check('V(TERMINAL) == 0', ag.get_value(TERMINAL_STATE) == 0)
    check('Policy legal', ag.get_policy(start) in mdp.get_possible_actions(start))
    _vi('small_grid', it=50); _vi('canyon', lr=-0.01, it=50)
    check('VI runs on small_grid + canyon', True)

def q2():
    print('\nQ2: Mission Profiles')
    from students import analysis
    for n in ('question2a', 'question2b', 'question2c', 'question2d', 'question2e'):
        ans = getattr(analysis, n)()
        if ans == 'NOT POSSIBLE':
            check(f'{n}()', False, 'still NOT POSSIBLE')
        elif isinstance(ans, tuple) and len(ans) == 3:
            d, ns, l = ans
            check(f'{n}() = {ans}', 0 <= d <= 1 and 0 <= ns <= 1 and isinstance(l, (int, float)))
        else:
            check(f'{n}()', False, f'need 3-tuple, got {type(ans).__name__}')

def q3():
    print('\nQ3: Q-Learning')
    mdp, env, ag = _ql('base_camp', eps=0.0, alpha=0.5, n=100)
    ag.update(mdp.get_start_state(), 'east', (1, 0), 1.0)
    q = ag.get_q_value(mdp.get_start_state(), 'east')
    check('update: Q(start,east) ≈ 0.5', abs(q - 0.5) < 1e-6, f'got {q}')
    check('V(TERMINAL) == 0', ag.get_value(TERMINAL_STATE) == 0)
    random.seed(42)
    _, e2, a2 = _ql('small_grid', eps=0.5, alpha=0.5, n=500)
    _quiet(_train, a2, e2, 500)
    v = a2.get_value((0, 0))
    check(f'V(start) > 0.1 after 500 ep', v > 0.1, f'V={v:.4f}')

def q4():
    print('\nQ4: Epsilon Greedy')
    from students.qlearning_agents import QLearningAgent
    acts = ('north', 'south', 'east', 'west')
    ag = QLearningAgent(action_fn=lambda s: acts, num_training=5000,
                        epsilon=0.5, alpha=0.5, gamma=0.9)
    ag.q_values[('s', 'north')] = 10.0
    random.seed(42)
    cnt = sum(1 for _ in range(2000) if ag.get_action('s') == 'north')
    r = cnt / 2000
    check(f'eps=0.5 P(north)={r:.3f} in [0.4,0.95]', 0.4 <= r <= 0.95)
    ag.epsilon = 0.0
    check('eps=0 => 100% north', all(ag.get_action('s') == 'north' for _ in range(100)))

def q5():
    print('\nQ5: Q-Learning on Mars')
    random.seed(42)
    mdp, env, ag = _ql('small_grid', disc=0.8, eps=0.05, alpha=0.2, n=2000)
    _quiet(_train, ag, env, 2000)
    ag.epsilon = 0.0
    wins = 0
    for _ in range(100):
        env.reset(); lr = 0.0
        for _ in range(100):
            if env.is_terminal(): break
            s = env.get_current_state(); a = ag.get_action(s)
            if a is None: break
            _, lr = env.do_action(a)
        if lr > 0: wins += 1
    check(f'Win rate >= 80%: {wins}/100', wins >= 80)

ALL = {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5}

def main():
    ap = argparse.ArgumentParser(description='Quick self-check (no grade)')
    ap.add_argument('-q', action='append', default=[], choices=list(ALL))
    args = ap.parse_args()
    print('\n  Mars Rover RL -- Student Self-Check\n  ' + '=' * 40)
    for qn in (args.q or list(ALL)):
        old_err = sys.stderr; sys.stderr = io.StringIO()
        try: ALL[qn]()
        except NotImplementedError as e:
            check(f'{qn}', False, f'stub: {str(e).replace("*** Method not implemented: ", "")}')
        except Exception as e:
            check(f'{qn}', False, f'{type(e).__name__}: {e}')
        finally: sys.stderr = old_err
    print(f'\n  Result: {_p} passed, {_f} failed')
    if _f == 0: print('  \U0001F680 All checks passed!')
    print('\n  Run  python autograder.py  for your actual grade.\n')

if __name__ == '__main__':
    main()
