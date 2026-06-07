"""
Umut Ardıl Murat
S028641
CS 451
"""

"""
students/qlearning_agents.py -- Q3, Q4, Q5 student file (Python 3.10+).

STUDENT FILE -- fill in every method marked ``*** YOUR CODE HERE ***``.
Do NOT modify the constructors or the provided query methods.

    By now Rover-7 has actually landed, and it is a very different
    situation from the one in Q1. Mission Control no longer has a
    clean, uploaded terrain map: the rover is in an unexplored
    patch of the Red Planet, the solar interference budget is
    whatever the weather gives it, and the only way to learn the
    world is to drive around in it. Every action returns a noisy
    observation of the next cell and the reward (or crater) you
    found there. That is the setting of reinforcement learning:
    learn an optimal policy purely from experience, without ever
    inspecting a transition table.

    In Q3 you will implement tabular Q-learning. Q4 is an
    epsilon-greedy exploration policy on top of the same Q-table.
    Q5 re-skins the same agent for Rover-7's own hyper-parameters
    and verifies it converges on Mars grids and the Crawler robot.
"""

import random
from typing import Any

from agents.learning_agents import ReinforcementAgent
from util import Counter, flip_coin, raise_not_defined


class QLearningAgent(ReinforcementAgent):
    """Tabular Q-learning agent that learns from raw experience.

    The agent stores one scalar per ``(state, action)`` pair in a
    :class:`util.Counter`. Because Counters default to 0 for missing
    keys, a brand-new agent effectively believes every action in
    every state is worth exactly zero -- a neutral, optimism-free
    prior that matches the standard textbook version of Q-learning.

    Training happens one step at a time through :meth:`update`,
    which is driven by the base :class:`ReinforcementAgent`
    training loop via ``observe_transition``. Query methods
    (``get_q_value``, ``get_value``, ``get_policy``, ``get_action``)
    let the GUI, the autograder, and the episodic loop inspect and
    use the current policy.
    """

    def __init__(self, **kwargs: Any):
        """Construct a Q-learning agent and seed an empty Q-table.

        All hyper-parameters (``alpha``, ``epsilon``, ``gamma``,
        ``num_training``, ``action_fn``) are forwarded to
        :class:`ReinforcementAgent`. Rover-7's own default tuning
        lives in :class:`RoverQLearningAgent` below.

        The line that initialises ``self.q_values`` is provided for
        you -- do not delete it, and do not rename it; the autograder
        will look for this exact attribute.
        """
        ReinforcementAgent.__init__(self, **kwargs)
        # PROVIDED: the Q-value table itself. Counter() returns 0.0
        # for any key it has not seen, which is the standard initial
        # Q-value for tabular Q-learning. Do not replace this with a
        # plain dict.
        self.q_values: Counter = Counter()

    # ==================================================================
    # ======================  STUDENT CODE BELOW  =====================
    # ==================================================================

    def get_q_value(self, state: Any, action: Any) -> float:
        """Return Q(state, action) from the agent's current Q-table.

        The table is ``self.q_values``, a :class:`util.Counter`
        keyed by ``(state, action)`` pairs. Because Counter reads
        missing keys as 0, a ``(state, action)`` pair the agent
        has never updated will return 0.0 -- the desired initial
        estimate for unseen state-action pairs in tabular
        Q-learning.

        The rest of the code should always read Q values through
        ``self.get_q_value`` rather than touching ``self.q_values``
        directly, so that any future subclass that changes the
        representation continues to work.
        """
        # Counter returns 0 for keys we havent seen yet which is what we want
        return self.q_values[(state, action)]

    def compute_value_from_q_values(self, state: Any) -> float:
        """Return V(state) = max_a Q(state, a) over the legal actions.

        If ``state`` has no legal actions at all -- which happens
        for the absorbing terminal state of the Mars grid world --
        the value is defined to be 0.0.

        Access Q-values through :meth:`get_q_value` rather than
        reading ``self.q_values`` directly so that the accessor
        contract is the only public way to read a Q-value.
        """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:
            return 0.0
        # V(s) = max Q(s,a) over all legal actions
        vals = []
        for a in actions:
            vals.append(self.get_q_value(state, a))
        return max(vals)

    def compute_action_from_q_values(self, state: Any) -> Any:
        """Return the greedy action at ``state`` under the Q-table.

        This is the ``argmax_a Q(state, a)`` used by both
        :meth:`get_policy` and the non-exploratory branch of
        :meth:`get_action`. Two subtleties:

        1. If ``state`` has no legal actions (terminal state),
           return ``None``.

        2. **Break ties at random using** :func:`random.choice`.
           Deterministic tie breaking (always returning the first
           action among equals) causes a Q-learning agent that
           starts from an all-zero Q-table to pick the same action
           every time, which prevents it from ever exploring the
           others.

        Also note: because unseen actions have Q-value 0.0 (Counter
        default), if every action the agent has tried so far has a
        *negative* Q-value, an unseen action is implicitly the best
        -- include unseen actions in your argmax by reading Q
        through :meth:`get_q_value`.
        """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:
            return None

        # find the max Q value first
        best_val = self.get_q_value(state, actions[0])
        for a in actions:
            q = self.get_q_value(state, a)
            if q > best_val:
                best_val = q

        # collect all actions that tie for the best and pick randomly
        # (need random tie-breaking otherwise agent gets stuck doing the same thing)
        tied = []
        for a in actions:
            if self.get_q_value(state, a) == best_val:
                tied.append(a)
        return random.choice(tied)

    def get_action(self, state: Any) -> Any:
        """Choose the next action from ``state`` (epsilon-greedy).

        With probability ``self.epsilon`` take a random legal
        action (explore); otherwise follow the current greedy
        policy (exploit). Use :func:`util.flip_coin` to flip the
        exploration coin and :func:`random.choice` to pick the
        random action when the coin comes up heads.

        If ``state`` has no legal actions, return ``None`` instead
        of trying to sample from an empty list.
        """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:
            return None
        # epsilon-greedy: explore with prob epsilon, exploit otherwise
        if flip_coin(self.epsilon):
            # explore - pick any random action
            return random.choice(actions)
        # exploit - go with the best action we know
        return self.compute_action_from_q_values(state)

    def update(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        reward: float,
    ) -> None:
        """Apply the one-step Q-learning TD update for a transition.

        The classical update is::

            Q(s, a) <- Q(s, a) + alpha * [ r
                                           + gamma * max_a' Q(s', a')
                                           - Q(s, a) ]

        which you may equivalently write as a convex combination::

            Q(s, a) <- (1 - alpha) * Q(s, a)
                     + alpha * (r + gamma * V(s'))

        Use ``self.alpha`` for the learning rate, ``self.discount``
        for gamma, and :meth:`compute_value_from_q_values` to read
        off V(next_state) -- that helper already returns 0.0 for
        terminal states, so you do not need to special-case them.

        Update ``self.q_values`` in place; do not construct a new
        Counter.
        """
        # Q-learning update: Q(s,a) = (1-alpha)*Q(s,a) + alpha*(r + gamma*maxQ(s'))
        old_q = self.get_q_value(state, action)
        future_val = self.compute_value_from_q_values(next_state)
        sample = reward + self.discount * future_val
        self.q_values[(state, action)] = old_q + self.alpha * (sample - old_q)

    # ==================================================================
    # =====================  PROVIDED CODE BELOW  =====================
    # ==================================================================

    def get_policy(self, state: Any) -> Any:
        """Return the greedy action at ``state`` under the Q-values."""
        return self.compute_action_from_q_values(state)

    def get_value(self, state: Any) -> float:
        """Return V(state) = max_a Q(state, a) under the Q-values."""
        return self.compute_value_from_q_values(state)

    def final(self, state: Any) -> None:
        """End-of-episode hook; base class no-op."""
        pass


class RoverQLearningAgent(QLearningAgent):
    """Q-learning agent with hyper-parameters tuned for Rover-7.

    This is the agent the GUI instantiates when you run the Mars
    rover demo: the defaults are the ones that give Rover-7 a
    reasonable exploration budget on the grids shipped with this
    project. There is no code for you to write here; the class is
    nothing more than :class:`QLearningAgent` with different
    default values. Constructing it with explicit keyword arguments
    overrides any subset of the defaults.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        gamma: float = 0.8,
        alpha: float = 0.2,
        num_training: int = 0,
        **kwargs: Any,
    ):
        # Fold the tuned defaults into kwargs then forward. This
        # pattern lets callers override any subset without tripping
        # Python's "multiple values for keyword argument" error.
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['num_training'] = num_training
        QLearningAgent.__init__(self, **kwargs)
