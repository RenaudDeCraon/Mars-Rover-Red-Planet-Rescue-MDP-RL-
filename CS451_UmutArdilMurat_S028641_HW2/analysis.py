"""
Umut Ardıl Murat
S028641
CS 451
"""

"""
students/analysis.py -- Q2 student file (Python 3.10+).

STUDENT FILE -- you fill in the five policy-profile functions below.

    === Mission Control Policy Briefing ===

    The Canyon Grid is a narrow strip of terrain between two
    mission-critical objectives and a very long drop. At the top of
    the canyon wall there is a *close* extraction point worth +1
    reward -- think of it as a handful of easily-grabbed surface
    samples on a ledge just above the rover. Further along and
    harder to reach, the *distant* sample depot sits at +10: a
    much richer payload, but the rover has to traverse more of the
    canyon to get there. Running along the bottom of the whole map
    is a crater cliff (a row of -10 tiles) -- one wrong step down
    there ends the mission.

    Depending on how you tune the planner's knobs, value iteration
    will produce dramatically different policies: a risk-tolerant
    cliff run, a safe upper detour, a "stall forever" policy that
    refuses to commit to any exit, and everything in between.

    Your job in Q2 is to find, for each of five distinct behaviours
    Mission Control wants to see, a setting of three parameters
    that produces that behaviour when you run
    :class:`students.value_iteration_agents.ValueIterationAgent`
    on the ``canyon`` grid. The knobs you can turn are:

    * ``discount``       -- the planner's gamma. Small values
      (close to 0) make the rover myopic and reluctant to walk
      far for a bigger reward; large values (close to 1) reward
      patient travel.
    * ``noise``          -- the slip probability that the Mars
      grid uses for stochastic transitions. With ``noise == 0``
      the rover can hug the cliff and never actually fall off;
      with non-zero noise the cliff becomes genuinely dangerous
      because any step along it might slip downward.
    * ``living_reward``  -- the reward earned on every
      non-extraction step. Negative values punish dawdling;
      positive values reward it (and if the bonus is big enough,
      can tip the optimal policy into refusing to ever terminate).

Each of the five functions should return a tuple
``(discount, noise, living_reward)`` that makes value iteration
pick the indicated behaviour on the Canyon Grid. If you decide a
particular combination cannot be achieved with any setting of the
three parameters, return the literal string ``'NOT POSSIBLE'``
instead -- that is the signal the autograder recognises.

Every stub currently returns ``'NOT POSSIBLE'``; change the body
of each function to your chosen tuple once you have worked out
the tuning. You may find it helpful to try a setting in the GUI
first::

    python mars_rover.py -a value -i 100 -g canyon \\
        --discount 0.9 --noise 0.2 --living-reward -0.1

and only commit to a final tuple once the arrows on screen match
the behaviour described.
"""


def question2a() -> tuple[float, float, float] | str:
    """Prefer the CLOSE +1 exit, and take the RISKY cliffside route.

    Mission Control wants the rover to grab the quick +1 ledge
    samples and to race there along the bottom of the canyon,
    right next to the crater cliff. "Risky" means the agent
    should be willing to walk along tiles adjacent to the -10
    cliff row; "close exit" means the greedy policy should take
    the +1 and not the +10.

    Return a ``(discount, noise, living_reward)`` tuple that
    produces that policy.
    """
    # low discount -> prefers close +1 over distant +10
    # no noise -> cliff is safe so it takes the short risky path
    return (0.2, 0.0, 0.0)


def question2b() -> tuple[float, float, float] | str:
    """Prefer the CLOSE +1 exit, but AVOID the cliff.

    Same goal as (a) -- head for the +1 ledge rather than the
    +10 depot -- but now the rover should take the safe route
    along the upper part of the canyon instead of hugging the
    crater cliff.

    Return a ``(discount, noise, living_reward)`` tuple.
    """
    # still low discount so it goes for close exit
    # but now noise is on, so the cliff is dangerous -> takes safe upper path
    return (0.2, 0.2, 0.0)


def question2c() -> tuple[float, float, float] | str:
    """Prefer the DISTANT +10 depot, and take the RISKY cliffside route.

    The rover should walk the whole length of the canyon to
    reach the +10 sample depot, and it should do so along the
    bottom (risking the cliff) rather than along the top.

    Return a ``(discount, noise, living_reward)`` tuple.
    """
    # high discount -> values the +10 enough to walk all the way there
    # no noise -> safe to hug the cliff, takes bottom path
    return (0.9, 0.0, 0.0)


def question2d() -> tuple[float, float, float] | str:
    """Prefer the DISTANT +10 depot, but AVOID the cliff.

    This is the "ideal mission profile": go for the high-value
    depot, but take the safe detour that stays well away from
    the crater cliff.

    Return a ``(discount, noise, living_reward)`` tuple.
    """
    # high discount to go for +10, noise on so cliff is scary -> avoids it
    return (0.9, 0.2, 0.0)


def question2e() -> tuple[float, float, float] | str:
    """Avoid every exit AND the crater cliff forever.

    In this mode the rover should refuse to extract at either
    reward cell and should never step onto the crater cliff.
    Because the MDP has no other absorbing states, achieving
    this means finding parameters for which the optimal policy
    would rather wander the map indefinitely than commit to any
    terminal. (Hint: the sign of the living reward matters
    here.)

    Return a ``(discount, noise, living_reward)`` tuple.
    """
    # positive living reward means the rover prefers to just keep wandering
    # since it gets +1 every step, no exit can beat infinite steps
    return (0.9, 0.2, 1.0)


if __name__ == '__main__':
    # Running this file directly prints the current answers so you
    # can eyeball them before submitting. The autograder reads the
    # functions directly, not this block.
    print("Mission Control Policy Briefing -- current answers:")
    for name in ('question2a', 'question2b', 'question2c', 'question2d', 'question2e'):
        answer = globals()[name]()
        print(f"  {name}(): {answer}")
