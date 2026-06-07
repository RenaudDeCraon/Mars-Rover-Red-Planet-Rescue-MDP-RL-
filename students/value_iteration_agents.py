"""
Umut Ardıl Murat
S028641
CS 451
"""

"""
students/value_iteration_agents.py -- Q1 student file (Python 3.10+).

STUDENT FILE -- you fill in the three methods marked
``*** YOUR CODE HERE ***``. Do NOT modify the constructor or the
four provided query methods at the bottom of the class.

    Mission Control has provided complete terrain maps for the
    region Rover-7 is about to cross. That means you, the flight
    software engineer, know the full Markov decision process before
    the rover touches down: every state, every legal action, the
    probability that the solar wind will nudge a drive command
    sideways, and the reward (or penalty) for each possible
    outcome. Your job is to plan ahead -- compute an optimal
    policy *before* deployment so that when Rover-7 wakes up on
    the surface it already knows what to do from every square on
    the map.

    In Q1 you will implement tabular value iteration, the classic
    planning algorithm that does exactly that: it repeatedly sweeps
    the state space, updating each state's value from the values
    of its successors under the current estimate. After a few
    dozen sweeps the values converge and the greedy policy with
    respect to those values is optimal.

You will fill in three methods below. Read each docstring carefully
-- it describes both *what* to compute and *which* pieces of the
MDP interface (``get_transition_states_and_probs``, ``get_reward``)
fit together to make it happen.
"""

from typing import Any

from agents.learning_agents import ValueEstimationAgent
from util import Counter, raise_not_defined


class ValueIterationAgent(ValueEstimationAgent):
    """A planner that runs batch value iteration on a fully-known MDP.

    Mission Control hands you a :class:`env.mdp.MarkovDecisionProcess`
    describing the planet surface. Because the model is known, you
    can compute the optimal value function V*(s) offline -- no
    experience required -- and then, at query time, read off the
    greedy action at any state as ``pi*(s) = argmax_a Q*(s, a)``.

    The agent stores its current estimate of V in ``self.values``, a
    :class:`util.Counter` mapping each state to a scalar. States
    the agent has not yet touched read as 0 by virtue of Counter's
    missing-key default -- exactly the right initial value for a
    value iteration sweep.

    Attributes
    ----------
    mdp : env.mdp.MarkovDecisionProcess
        The fully-known planning problem Mission Control uploaded.
    discount : float
        Discount factor gamma applied to future rewards.
    iterations : int
        Number of Bellman-backup sweeps to run at construction time.
    values : util.Counter
        The current estimate of V; indexed by state, defaults to 0.
    """

    def __init__(
        self,
        mdp: Any,
        discount: float = 0.9,
        iterations: int = 100,
    ):
        """Stash the planning problem and run value iteration offline.

        The whole point of a model-based planner is that all the
        expensive work happens *before* the rover is deployed. We
        therefore run the full batch of iterations right here in the
        constructor so the resulting agent is ready to answer
        ``get_action`` queries instantly.
        """
        # Pull the ValueEstimationAgent defaults into place so that
        # callers can still read self.alpha/self.epsilon if they want
        # to; for a planner they are unused, but keeping them around
        # makes the agent interchangeable with learning agents.
        super().__init__(gamma=discount)

        self.mdp = mdp
        self.discount: float = float(discount)
        self.iterations: int = int(iterations)
        # Counter's missing-key-returns-zero behaviour is exactly the
        # right initial value function for value iteration: every
        # state starts at V_0(s) = 0.
        self.values: Counter = Counter()

        # Run planning now so the agent is fully prepared at
        # construction time. Do NOT defer this to first query --
        # ``get_action`` must be able to answer immediately.
        self.run_value_iteration()

    # ==================================================================
    # ======================  STUDENT CODE BELOW  =====================
    # ==================================================================

    def run_value_iteration(self) -> None:
        """Run ``self.iterations`` sweeps of batch value iteration.

        For Q1 you will implement the classic batch Bellman backup::

            for k in range(iterations):
                new_values = Counter()
                for each non-terminal state s with at least one legal
                    action a:
                    new_values[s] = max_a Q_k(s, a)
                self.values = new_values

        The critical detail is the word **batch**: every V_{k+1}(s)
        must be computed from the *old* V_k, not from values that
        have already been updated in the current sweep. Concretely,
        create a fresh :class:`util.Counter`, fill it in using the
        current ``self.values`` to look up successor values, and
        only after the sweep is finished replace ``self.values``
        with the new counter.

        Terminal states -- the ones for which
        ``self.mdp.is_terminal`` returns ``True`` -- keep a value
        of 0 and should simply be skipped. A state that
        ``mdp.get_possible_actions`` reports as having no legal
        actions should also be left at 0.

        Hint: you already have a helper for the inner max -- use
        :meth:`compute_q_value_from_values` once you implement it
        below, and the body of this method becomes a two-line loop.
        """
        # run k iterations of bellman backup
        for i in range(self.iterations):
            # important: batch update, so we need a NEW counter each time
            updated = Counter()
            for s in self.mdp.get_states():
                # skip terminal states, they stay at 0
                if self.mdp.is_terminal(s):
                    continue
                possible = self.mdp.get_possible_actions(s)
                if len(possible) == 0:
                    continue
                # V(s) = max over all actions of Q(s,a)
                best_q = float('-inf')
                for a in possible:
                    q = self.compute_q_value_from_values(s, a)
                    if q > best_q:
                        best_q = q
                updated[s] = best_q
            # only swap after the full sweep is done
            self.values = updated

    def compute_q_value_from_values(self, state: Any, action: Any) -> float:
        """Return Q(state, action) computed from the current ``self.values``.

        The Bellman equation for Q under a known model is::

            Q(s, a) = sum_{s'} T(s, a, s') * [ R(s, a, s')
                                               + gamma * V(s') ]

        where the sum ranges over all next states ``s'`` with
        non-zero transition probability. Use the MDP interface to
        look up both pieces:

        * ``self.mdp.get_transition_states_and_probs(state, action)``
          returns a list of ``(next_state, probability)`` pairs.
        * ``self.mdp.get_reward(state, action, next_state)`` returns
          the reward for that specific transition.

        The discount factor is in ``self.discount``. Values come
        from ``self.values`` (which you should *read*, not modify,
        in this method -- the batch update in
        :meth:`run_value_iteration` relies on this being a pure
        function of the current values).
        """
        # Q(s,a) = sum over s' of T(s,a,s') * [R(s,a,s') + gamma * V(s')]
        total = 0.0
        transitions = self.mdp.get_transition_states_and_probs(state, action)
        for next_state, prob in transitions:
            r = self.mdp.get_reward(state, action, next_state)
            total += prob * (r + self.discount * self.values[next_state])
        return total

    def compute_action_from_values(self, state: Any) -> Any:
        """Return the greedy action at ``state`` under ``self.values``.

        The policy is ``pi(s) = argmax_a Q(s, a)``. Use
        :meth:`compute_q_value_from_values` to score each legal
        action and return the best one. If the state has no legal
        actions (for example, a terminal state) return ``None``.

        Ties can be broken arbitrarily; the autograder does not
        care which of several equally-valued actions you return,
        only that the Q-value of the returned action is a maximum.
        """
        possible = self.mdp.get_possible_actions(state)
        if len(possible) == 0:
            return None
        # argmax: find action with highest Q value
        best_act = None
        best_val = float('-inf')
        for a in possible:
            qval = self.compute_q_value_from_values(state, a)
            if qval > best_val:
                best_val = qval
                best_act = a
        return best_act

    # ==================================================================
    # =====================  PROVIDED CODE BELOW  =====================
    # ==================================================================
    #
    # Do not modify anything past this line. These four methods are
    # thin wrappers that let the rest of the project (the GUI, the
    # autograder, the training loop) query your agent through the
    # standard ValueEstimationAgent interface.

    def get_value(self, state: Any) -> float:
        """Return the current estimate of V(state)."""
        return self.values[state]

    def get_q_value(self, state: Any, action: Any) -> float:
        """Return Q(state, action) under the current value function."""
        return self.compute_q_value_from_values(state, action)

    def get_policy(self, state: Any) -> Any:
        """Return the greedy action at ``state`` under current values."""
        return self.compute_action_from_values(state)

    def get_action(self, state: Any) -> Any:
        """Return the action the planner would take at ``state``.

        For a deterministic planner this is identical to
        :meth:`get_policy` -- there is no exploration to add on
        top, the planner always takes its best guess.
        """
        return self.compute_action_from_values(state)
