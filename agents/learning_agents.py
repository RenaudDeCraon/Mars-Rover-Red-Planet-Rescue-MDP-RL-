"""
agents/learning_agents.py -- abstract RL agent base classes (Python 3.10+).

Part of the "Mars Rover: Red Planet Rescue" homework for CS 451/551.

Two abstract agents live here:

* :class:`ValueEstimationAgent` -- anything that can, on demand,
  answer "what is Q(s, a)?", "what is V(s)?", "what is pi(s)?",
  and "what action should I take from s right now?". Value
  iteration (a planner with full access to the MDP) and Q-learning
  (a model-free learner) both satisfy this interface, which lets
  the rest of the project treat them interchangeably for display
  and evaluation.
* :class:`ReinforcementAgent` -- a :class:`ValueEstimationAgent`
  that also experiences episodes: it is driven by an environment,
  observes transitions, accumulates rewards, and applies a learning
  rule via :meth:`ReinforcementAgent.update`. It handles the
  bookkeeping that every episodic RL agent needs (episode
  counting, train/test split, exploration decay, progress
  reporting), leaving subclasses to focus on the one-line TD
  update that defines their algorithm.

Neither class implements a concrete learning rule; the student
files ``students/value_iteration_agents.py`` and
``students/qlearning_agents.py`` fill those in.
"""

from typing import Any, Callable

from util import raise_not_defined


class ValueEstimationAgent:
    """Abstract base for agents that expose Q- and V-estimates on demand.

    The contract is four questions a caller can ask an instance:
    :meth:`get_q_value`, :meth:`get_value`, :meth:`get_policy`, and
    :meth:`get_action`. The first three report the agent's current
    estimate of Q*, V*, and pi* for the requested state; the fourth
    is "what do you *actually* want to do next?", which for an
    epsilon-greedy learner adds exploration on top of the policy.

    The constructor takes the four hyper-parameters common to nearly
    every tabular RL algorithm: learning rate ``alpha``, exploration
    rate ``epsilon``, discount factor ``gamma`` (stored as
    ``self.discount``), and an episode budget ``num_training``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        epsilon: float = 0.05,
        gamma: float = 0.8,
        num_training: int = 10,
    ):
        # Cast to float so downstream arithmetic is never surprised
        # by Python's integer division or a mis-typed CLI argument.
        self.alpha: float = float(alpha)
        self.epsilon: float = float(epsilon)
        self.discount: float = float(gamma)
        self.num_training: int = int(num_training)

    # ------------------------------------------------------------------
    # Abstract interface -- subclasses must override all four.
    # ------------------------------------------------------------------

    def get_q_value(self, state: Any, action: Any) -> float:
        """Return the agent's estimate of Q*(state, action)."""
        raise_not_defined()

    def get_value(self, state: Any) -> float:
        """Return the agent's estimate of V*(state)."""
        raise_not_defined()

    def get_policy(self, state: Any) -> Any:
        """Return the greedy action at ``state`` under current estimates."""
        raise_not_defined()

    def get_action(self, state: Any) -> Any:
        """Return the action the agent wants to take next from ``state``.

        Learning agents typically add epsilon-greedy exploration on
        top of :meth:`get_policy`; pure planners usually just
        delegate.
        """
        raise_not_defined()


class ReinforcementAgent(ValueEstimationAgent):
    """Abstract base for episodic reinforcement-learning agents.

    A :class:`ReinforcementAgent` lives inside a training loop that
    resets the environment between episodes, repeatedly asks the
    agent for an action, and pipes every resulting transition back
    into the agent via :meth:`observe_transition`. The base class
    handles all the bookkeeping:

    * ``episodes_so_far``, ``accum_train_rewards``, ``accum_test_rewards``
      -- the running statistics the autograder and GUI read.
    * ``episode_rewards`` -- the return of the episode currently in
      progress, reset by :meth:`start_episode` and rolled into the
      appropriate accumulator by :meth:`stop_episode`.
    * Exploration decay -- once ``episodes_so_far`` reaches
      ``num_training`` the agent flips to full exploitation by
      zeroing out both ``epsilon`` and ``alpha``. The boundary is
      logged with a check-mark banner so students can tell from
      the console that the train/test switch happened.
    * Progress reports -- every ``_REPORT_INTERVAL`` training
      episodes the average return over the most recent window is
      printed with a chart banner.

    Subclasses implement exactly one method: :meth:`update`, the
    TD update that defines the learning algorithm. Everything else
    in the training loop flows through the base class.
    """

    # Progress is reported every this-many training episodes.
    # Parameterised as a class attribute so the autograder can
    # silence or speed up the output by overriding it on a subclass.
    _REPORT_INTERVAL: int = 100

    def __init__(
        self,
        action_fn: Callable[[Any], Any] | None = None,
        num_training: int = 100,
        epsilon: float = 0.5,
        alpha: float = 0.5,
        gamma: float = 1.0,
    ):
        super().__init__(
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            num_training=num_training,
        )

        # ``action_fn`` lets the agent ask its environment "what are
        # the legal moves from state s?". The training loop typically
        # passes ``env.get_possible_actions``. When constructed
        # outside any environment (e.g. during unit testing), default
        # to an empty list so calls are well-defined.
        if action_fn is None:
            action_fn = lambda state: []
        self.action_fn: Callable[[Any], Any] = action_fn

        # Cross-episode counters. Train and test reward totals are
        # kept separate so the autograder can compute test-time
        # averages without having to subtract the training window.
        self.episodes_so_far: int = 0
        self.accum_train_rewards: float = 0.0
        self.accum_test_rewards: float = 0.0
        self.episode_rewards: float = 0.0

        # Snapshot of ``accum_train_rewards`` at the previous progress
        # report. Subtracting from the current total gives the return
        # earned in the most recent window -- which is what students
        # actually want to see while training.
        self._last_report_rewards: float = 0.0

        # Some training loops prefer to let the agent remember its
        # most recent transition instead of having the environment
        # thread it through observe_transition explicitly.
        self.last_state: Any = None
        self.last_action: Any = None

        # If the caller asked for zero training episodes we are
        # effectively in test mode from the very first step. Flip
        # the exploration knobs now so no exploratory action is
        # ever sampled.
        if self.num_training == 0:
            self.epsilon = 0.0
            self.alpha = 0.0

    # ------------------------------------------------------------------
    # Core learning hook -- the one thing subclasses must implement.
    # ------------------------------------------------------------------

    def update(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        reward: float,
    ) -> None:
        """Apply the agent's TD update for a single transition.

        Subclasses implement this method; for tabular Q-learning it
        is roughly::

            sample = reward + discount * max_a' Q(next_state, a')
            Q(s, a) = (1 - alpha) * Q(s, a) + alpha * sample
        """
        raise_not_defined()

    # ------------------------------------------------------------------
    # Environment-facing helpers.
    # ------------------------------------------------------------------

    def get_legal_actions(self, state: Any) -> Any:
        """Return the legal actions at ``state`` via ``self.action_fn``."""
        return self.action_fn(state)

    def observe_transition(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        delta_reward: float,
    ) -> None:
        """Record a transition and forward it to :meth:`update`.

        ``delta_reward`` is added to the running episode return
        *before* the subclass's update rule runs, matching Berkeley's
        convention. Accumulating first means a learner that peeks at
        ``self.episode_rewards`` inside :meth:`update` sees the
        reward it is about to learn from.
        """
        self.episode_rewards += delta_reward
        self.update(state, action, next_state, delta_reward)

    # ------------------------------------------------------------------
    # Episode lifecycle.
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """Reset per-episode state at the start of a new episode.

        The cross-episode counters (``episodes_so_far``,
        ``accum_*_rewards``) are deliberately *not* reset here --
        those are the whole point of this base class.
        """
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0

    def stop_episode(self) -> None:
        """Roll the finished episode's return into the right bucket.

        Also handles the two pieces of end-of-episode housekeeping:
        printing a progress report every ``_REPORT_INTERVAL``
        training episodes, and (once) flipping ``epsilon`` and
        ``alpha`` to zero at the train -> test boundary.
        """
        # Episodes that began during training count toward the
        # training total; everything after num_training counts as
        # test. We do the comparison *before* incrementing so the
        # Nth training episode lands in the training bucket rather
        # than spilling over into test.
        if self.episodes_so_far < self.num_training:
            self.accum_train_rewards += self.episode_rewards
        else:
            self.accum_test_rewards += self.episode_rewards
        self.episodes_so_far += 1

        # Progress report at every multiple of _REPORT_INTERVAL up
        # to and including num_training. We compare the current
        # accumulator to the snapshot taken at the previous report,
        # so the average reflects only the most recent window
        # rather than the cumulative mean.
        if (
            self.episodes_so_far % self._REPORT_INTERVAL == 0
            and self.episodes_so_far <= self.num_training
        ):
            window_total = self.accum_train_rewards - self._last_report_rewards
            window_avg = window_total / self._REPORT_INTERVAL
            print(
                f"\U0001F4CA Rover-7 training progress: "
                f"episode {self.episodes_so_far}/{self.num_training}, "
                f"avg reward over last {self._REPORT_INTERVAL} "
                f"episodes = {window_avg:.2f}"
            )
            self._last_report_rewards = self.accum_train_rewards

        # Train -> test boundary. Use == so the banner prints
        # exactly once, even if stop_episode is called again.
        if self.episodes_so_far == self.num_training:
            train_avg = (
                self.accum_train_rewards / self.num_training
                if self.num_training > 0
                else 0.0
            )
            print(
                f"\u2705 Training complete after {self.num_training} "
                f"episodes (avg train reward = {train_avg:.2f}). "
                f"Switching to exploitation: epsilon=0, alpha=0."
            )
            self.epsilon = 0.0
            self.alpha = 0.0

    # ------------------------------------------------------------------
    # Train/test status queries.
    # ------------------------------------------------------------------

    def is_in_training(self) -> bool:
        """True while the agent is still consuming its training budget."""
        return self.episodes_so_far < self.num_training

    def is_in_testing(self) -> bool:
        """True once the training budget has been exhausted."""
        return not self.is_in_training()
