# agents/ — Agent Base Classes

## Do Not Edit

Students should **NOT** modify files in this directory. These are
abstract base classes that the student implementations in
`students/` extend. Editing them will break the autograder's
import chain.

---

## Files

### learning_agents.py

Two abstract agent classes that define the interface every planner
and learner in the project must satisfy.

---

#### ValueEstimationAgent

**Purpose:** Base for any agent that can, on demand, answer "what
is Q(s, a)?", "what is V(s)?", "what is the greedy action?", and
"what action should I take next?". Both value iteration (a planner
with full MDP access) and Q-learning (a model-free learner) satisfy
this interface, which lets the rest of the project treat them
interchangeably for display, evaluation, and grading.

**Constructor:**

```python
ValueEstimationAgent(alpha=1.0, epsilon=0.05, gamma=0.8, num_training=10)
```

| Parameter | Stored as | Description |
|---|---|---|
| `alpha` | `self.alpha` | Learning rate (unused by planners) |
| `epsilon` | `self.epsilon` | Exploration rate (unused by planners) |
| `gamma` | `self.discount` | Discount factor |
| `num_training` | `self.num_training` | Training episode budget |

**Abstract methods** (all call `util.raise_not_defined()`):

| Method | Returns | Description |
|---|---|---|
| `get_q_value(state, action)` | `float` | Agent's estimate of Q*(s, a) |
| `get_value(state)` | `float` | Agent's estimate of V*(s) |
| `get_policy(state)` | `Any` | Greedy action under current estimates |
| `get_action(state)` | `Any` | Action the agent actually takes (may include exploration) |

**Extended by:** `ValueIterationAgent` (in `students/value_iteration_agents.py`)

---

#### ReinforcementAgent(ValueEstimationAgent)

**Purpose:** Base for agents that learn from experience — driven
by an environment, observing transitions one at a time. Handles
all the episodic bookkeeping (episode counting, train/test split,
exploration decay, progress reporting) so subclasses only implement
the one-line TD update that defines their algorithm.

**Constructor:**

```python
ReinforcementAgent(
    action_fn=None,        # callable: state → legal actions
    num_training=100,      # episodes before switching to exploitation
    epsilon=0.5,           # exploration rate
    alpha=0.5,             # learning rate
    gamma=1.0,             # discount factor
)
```

If `action_fn` is `None`, it defaults to `lambda state: []`.
If `num_training` is 0, `epsilon` and `alpha` are zeroed at
construction (immediate exploitation).

**Tracked attributes:**

| Attribute | Description |
|---|---|
| `episodes_so_far` | How many episodes have completed |
| `accum_train_rewards` | Total reward across all training episodes |
| `accum_test_rewards` | Total reward across all test episodes |
| `episode_rewards` | Reward accumulated in the current episode |
| `last_state`, `last_action` | Most recent transition (reset each episode) |

**Provided methods:**

| Method | Description |
|---|---|
| `get_legal_actions(state)` | Delegates to `self.action_fn(state)` |
| `observe_transition(state, action, next_state, delta_reward)` | Adds `delta_reward` to `episode_rewards`, then calls `self.update(...)` |
| `start_episode()` | Resets `last_state`, `last_action`, `episode_rewards` (NOT the cross-episode counters) |
| `stop_episode()` | Routes `episode_rewards` into the train or test accumulator; prints a progress report every 100 training episodes; at the train→test boundary, zeros `epsilon` and `alpha` and prints a completion banner |
| `is_in_training()` | `True` while `episodes_so_far < num_training` |
| `is_in_testing()` | `True` once the training budget is exhausted |

**Training → testing switch:**

After `num_training` episodes have completed (detected in
`stop_episode`), `epsilon` and `alpha` are set to `0.0`. This
means:
- The agent stops exploring (epsilon-greedy always picks the
  greedy action).
- The agent stops learning (Q-values are frozen).

This happens automatically — the student does not need to implement
it.

**Progress reporting:**

Every 100 training episodes, a chart emoji line is printed:
```
📊 Rover-7 training progress: episode 100/500, avg reward over last 100 episodes = 0.84
```

At the train→test boundary, a completion banner is printed:
```
✅ Training complete after 500 episodes (avg train reward = 0.91). Switching to exploitation: epsilon=0, alpha=0.
```

**Abstract method** (students implement in subclasses):

| Method | Description |
|---|---|
| `update(state, action, next_state, reward)` | The core learning rule (e.g., the Q-learning TD update) |

**Extended by:** `QLearningAgent`
(in `students/qlearning_agents.py`)

---

## Class Hierarchy

```
ValueEstimationAgent                  agents/learning_agents.py
│
├── ValueIterationAgent               students/value_iteration_agents.py
│     3 stubs: run_value_iteration,
│     compute_q_value_from_values,
│     compute_action_from_values
│
└── ReinforcementAgent                agents/learning_agents.py
    │  1 abstract: update
    │
    └── QLearningAgent                students/qlearning_agents.py
        │  5 stubs: get_q_value,
        │  compute_value_from_q_values,
        │  compute_action_from_q_values,
        │  get_action, update
        │
        └── RoverQLearningAgent       students/qlearning_agents.py
              (tuned defaults only,
               no student code)
```

---

## Import Example

```python
from agents.learning_agents import ValueEstimationAgent
from agents.learning_agents import ReinforcementAgent
```

Students see these in their files:

```python
# In students/value_iteration_agents.py:
from agents.learning_agents import ValueEstimationAgent

# In students/qlearning_agents.py:
from agents.learning_agents import ReinforcementAgent
```
