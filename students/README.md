# students/ — Student Implementation Files

## These Are the ONLY Files You Edit

You submit exactly these **3 files**. Do not modify anything else
in the project — the autograder imports your code from this
directory and grades it against reference solutions.

Every method you need to fill in is marked with
`*** YOUR CODE HERE ***` and calls `raise_not_defined()`, which
prints the exact file, line number, and method name so you can
find it immediately.

---

## Files

### value_iteration_agents.py — Question 1 (20 points)

**Class:** `ValueIterationAgent`
**Extends:** `ValueEstimationAgent` (from `agents/learning_agents.py`)

**Methods to implement:**

| # | Method | What it does |
|---|---|---|
| 1 | `run_value_iteration()` | Run `self.iterations` sweeps of batch value iteration. For each sweep, create a **fresh** `Counter`, compute V_{k+1}(s) = max_a Q_k(s,a) for every non-terminal state using the *old* values, then replace `self.values` with the new counter. |
| 2 | `compute_q_value_from_values(state, action)` | Compute Q(s,a) = sum_{s'} T(s,a,s') * [R(s,a,s') + gamma * V(s')] using `self.mdp.get_transition_states_and_probs` and `self.mdp.get_reward`. |
| 3 | `compute_action_from_values(state)` | Return the greedy action argmax_a Q(s,a). Return `None` if the state has no legal actions. Ties can be broken arbitrarily. |

**Already provided (do not modify):**
`get_value`, `get_q_value`, `get_policy`, `get_action` — thin
wrappers that delegate to your implementations above.

**Common mistakes:**
- Using **in-place** updates instead of **batch**: if you update
  `self.values[s]` inside the loop instead of writing to a
  separate `new_values` counter and swapping at the end, earlier
  updates pollute later computations in the same sweep.
- Forgetting that **terminal states** have no legal actions and
  should be skipped (their value stays at 0).
- Not using `compute_q_value_from_values` inside
  `run_value_iteration` — the docstring says "Hint: you already
  have a helper for the inner max."

---

### qlearning_agents.py — Questions 3, 4, 5 (35 points)

Two classes live here. You only write code in one of them.

#### QLearningAgent (Q3: 20 pts, Q4: 8 pts, Q5: 7 pts)

**Extends:** `ReinforcementAgent` (from `agents/learning_agents.py`)

**Methods to implement:**

| # | Method | What it does |
|---|---|---|
| 1 | `get_q_value(state, action)` | Return `self.q_values[(state, action)]`. Counter returns 0 for unseen pairs. |
| 2 | `compute_value_from_q_values(state)` | Return max_a Q(s,a) over legal actions. Return 0.0 if no legal actions (terminal). **Must go through `self.get_q_value()`**, not `self.q_values` directly — keep the accessor as the single public read path. |
| 3 | `compute_action_from_q_values(state)` | Return the greedy action. **Break ties randomly** with `random.choice`. Return `None` if no legal actions. |
| 4 | `get_action(state)` | Epsilon-greedy: with probability `self.epsilon`, return `random.choice(legal_actions)`. Otherwise return the greedy action. Use `util.flip_coin(self.epsilon)`. Return `None` if no legal actions. |
| 5 | `update(state, action, next_state, reward)` | Q(s,a) = (1-alpha) * Q(s,a) + alpha * (reward + gamma * V(s')). Use `self.discount` for gamma and `compute_value_from_q_values` for V(s'). |

#### RoverQLearningAgent (Q5 — no code to write)

Just `QLearningAgent` with tuned defaults: `epsilon=0.05`,
`gamma=0.8`, `alpha=0.2`, `num_training=0`. You don't edit this
class.

**Already provided (do not modify):**
`RoverQLearningAgent` (defaults only), `get_policy`, `get_value`,
`final`.

**Common mistakes:**
- **Not breaking ties randomly** in `compute_action_from_q_values`:
  with a deterministic tie-break (e.g., always returning the first
  action), the agent picks the same direction from an all-zero
  Q-table every episode, preventing exploration of alternatives.
- **Forgetting that unseen actions have Q = 0**: if every action
  the agent has tried has a *negative* Q-value, an unseen action
  (Q = 0 via the Counter default) is actually the best. You must
  include unseen actions in your argmax — read Q through
  `self.get_q_value`, not by iterating `self.q_values`.
- **Accessing `self.q_values` directly** in `compute_value_from_q_values`
  instead of going through `self.get_q_value()`: keep the accessor
  as the single public read path so future representation changes
  don't break this method.

---

### analysis.py — Question 2 (20 points)

**Functions:** `question2a` through `question2e`

Each returns a `(discount, noise, living_reward)` tuple — or the
string `'NOT POSSIBLE'` if you believe the behaviour cannot be
achieved.

The **Canyon Grid** has:
- A close exit worth **+1** at the top-right
- A distant sample depot worth **+10** at the middle-right
- A crater cliff of **-10** tiles along the bottom row

The three knobs you tune:

| Knob | Effect |
|---|---|
| `discount` (gamma) | Low → myopic (prefer nearby rewards). High → patient (walk far for bigger payoff). |
| `noise` | 0 → deterministic (can safely hug the cliff). > 0 → cliff is dangerous (might slip into it). |
| `living_reward` | Negative → punishes dawdling (hurry to any exit). Positive → rewards wandering (may refuse to terminate). |

**The five profiles:**

| Function | Behaviour | Hints |
|---|---|---|
| `question2a` | Close +1 exit, **risky** cliffside route | Low discount (myopic), no noise (cliff safe), negative living reward |
| `question2b` | Close +1 exit, **safe** upper route | Low discount, some noise (cliff dangerous), negative living reward |
| `question2c` | Distant +10 depot, **risky** cliffside route | High discount (patient), no noise, small negative living reward |
| `question2d` | Distant +10 depot, **safe** upper route | High discount, some noise, small negative living reward |
| `question2e` | **Never terminate** — avoid all exits and craters forever | High discount + positive living reward → infinite wandering has higher value than any terminal |

---

## How to Test Your Code

```bash
# Quick self-check (no grade — just ✅ / ❌ per check)
python check.py
python check.py -q q1      # check just Q1

# Full autograder (75 points)
python autograder.py
python autograder.py -q q1  # grade just Q1

# Interactive exploration
python mars_rover.py -a value -g base_camp -i 100 -k 5   # watch VI policy
python mars_rover.py -a q -g canyon -k 200                # watch QL learn
python mars_rover.py -m                                   # manual keyboard
```

---

## Recommended Order

```
Q1 (Value Iteration)  →  easiest; get the planner working first
Q2 (Analysis)         →  uses your Q1 planner to find the right knobs
Q3 (Q-Learning)       →  the main model-free algorithm
Q4 (Epsilon-Greedy)   →  just the exploration policy on top of Q3
Q5 (Deployment)       →  run Q3 on Mars grids + crawler, verify convergence
```

Start with Q1 — if your value iteration works, the rest of the
project builds on top of it.
