# display/ — Visualization

## Purpose

Rendering the Mars Grid World and the Crawler robot. All display
code is **optional** — the project works fully in text mode without
Pygame. The autograder and `check.py` never use the GUI; they
suppress stdout during test execution and only print pass/fail
results.

---

## Files

### gui_display.py (400 lines)

**Class:** `MarsDisplay`
**Requires:** pygame (imported conditionally with `try/except`)

An animated Pygame visualisation of a `MarsGrid` with a
Mars-themed colour palette (dark reds, browns, golds). Every frame
is rendered at 60 Hz.

**Visual features:**

| Element | Description |
|---|---|
| Sky | Vertical gradient from dark purple-blue at the top to rusty orange at the horizon, with 100 twinkling stars |
| Terrain | Per-cell rock texture: base colour `#6a3e28` with random shade variation and 3–8 rock dots (deterministic via seed 42) |
| Walls | 3D effect: darker base, lighter top face, highlight edge, horizontal rock-texture lines |
| Sample sites (+) | Pulsing green glow overlay (sine wave), diamond crystal shape with inner shine, sparkle particles every 20 frames |
| Craters (-) | Pulsing red glow overlay, concentric crater rings (outer, dark inner, lava-glow centre), lava-ember particles every 30 frames |
| Rover | Yellow body with accent border, top panel, two blue solar panels with strut lines, vertical antenna with blinking red tip (every 20 frames), camera eye (dark circle + light blue iris + white highlight), "7" label, two wheels with rotating spokes (`frame * 0.15` angle), ground shadow ellipse, yellow glow underneath, bobbing animation (`sin(frame * 0.1) * 2` pixels) |
| Dust particles | Brown/tan, emitted from rover every 15 frames, upward velocity, gravity `vy += 0.04`, alpha fade by `life / max_life` |

**Overlay modes** (toggle with keyboard):

| Mode | Key | What it shows |
|---|---|---|
| Values | `V` | `V(s)` at each open cell, green if positive, red if negative |
| Q-values | `Q` | Cell split into 4 quarters by diagonal lines, Q-value in each (N/S/E/W) |
| Policy | `P` | Glowing gold arrow with triangular head; gold ring for `'extract'` |
| Grid | — | Terrain only, no agent info |

**HUD panel** (bottom of window):

Dark background with horizontal scanline effect, mini rover icon,
"ROVER-7 TELEMETRY" title in gold, "MARS EXPLORATION AGENCY" subtitle,
telemetry lines (MODE, POSITION, STEP, ACTION, REWARD with colour
coding: green for positive, red for negative), 5 animated signal
bars, and a controls help line.

**Key methods:**

| Method | Description |
|---|---|
| `draw(**kwargs)` | Render one frame. Accepts `env`, `state`, `agent`, `mode`, `step`, `action`, `reward`, `next_state`. Unknown kwargs silently ignored. |
| `handle_events()` | Pump pygame events. Returns an action string (`'north'`, `'south'`, `'east'`, `'west'`, `'extract'`) on arrow/space/enter press, or `None`. V/Q/P keys switch the overlay mode. ESC sets `self.should_quit`. |
| `wait_for_key_action(legal_actions)` | Loop on `handle_events()` until a legal action is pressed or the user quits. Used by manual mode. |
| `wait_for_key()` | Block until any key is pressed. |
| `close()` | Call `pygame.quit()`. |

**Keyboard controls:**

| Key | Action |
|---|---|
| Arrow keys | Move the rover (manual mode) |
| Space / Enter | Extract (manual mode) |
| V | Switch to values overlay |
| Q | Switch to Q-values overlay |
| P | Switch to policy overlay |
| ESC | Quit |

**Particle system:**

`_Particle` uses `__slots__` for minimal memory. `ParticleSystem`
manages a flat list with `emit_dust`, `emit_sparkle`, and
`emit_lava` methods. `update_and_draw` applies gravity, alpha
fading, and dead-particle removal each frame.

**Layout constants:**

| Constant | Value | Description |
|---|---|---|
| `CELL_SIZE` | 110 px | Width and height of each grid cell |
| `MARGIN_L`, `MARGIN_R` | 50 px | Left and right margins |
| `MARGIN_T` | 80 px | Top margin (for title bar) |
| `HUD_H` | 130 px | Height of the HUD panel |

Window size = `(50 + width×110 + 50, 80 + height×110 + 130)`.
For base_camp (4×3): 540×540. For canyon (5×4): 650×620.

---

### text_display.py (284 lines)

**Class:** `TextMarsDisplay`
**Requires:** nothing (no pygame dependency)

ASCII terminal renderer using Unicode box-drawing characters
(`┌ ─ ┬ ┐ │ ├ ┼ ┤ └ ┴ ┘`). Each cell is 6 characters wide.

**Cell content** (by priority):

| Priority | Content | Display |
|---|---|---|
| 1 (highest) | Wall | `######` |
| 2 | Rover position | `  R   ` |
| 3 | Terminal reward | ` +1  ` or ` -10 ` |
| 4 | Agent value/policy | `+0.66 ` or `  ↑   ` |
| 5 (lowest) | Open terrain | `      ` (blank) |

**Modes:** `'values'` (V(s) numbers), `'policy'` (Unicode arrows
↑↓←→ and `*` for extract), `'grid'` (terrain only).

**Key methods:**

| Method | Description |
|---|---|
| `draw(**kwargs)` | Print the grid to stdout. Accepts `env`, `state`, `agent`, `mode`, `header`, `footer`, plus training metadata (`step`, `action`, `reward`, `next_state`). |
| `handle_events()` | Returns `None` (no async input in text mode). |
| `wait_for_key()` | Prints "[press Enter to continue]" and blocks on `stdin.readline()`. |
| `close()` | No-op (no resources to release). |

**Example output:**

```
┌──────┬──────┬──────┬──────┐
│+0.50 │+0.44 │  R   │  -1  │
├──────┼──────┼──────┼──────┤
│+0.58 │######│+0.77 │+0.87 │
├──────┼──────┼──────┼──────┤
│+0.66 │+0.76 │+0.87 │  +1  │
└──────┴──────┴──────┴──────┘
```

This is the automatic fallback when Pygame is not installed.

---

### crawler_gui.py (100 lines)

**Function:** `run_crawler_gui(env, agent, steps=500, delay=0.0)`
**Requires:** pygame (imported conditionally)

An 800×450 Pygame window showing the crawler learning in real time.

**Visual elements:**

| Element | Description |
|---|---|
| Background | Dark Mars night `(25, 15, 12)` |
| Ground | Rusty rectangle `(120, 70, 45)` below y=300 with a horizon stripe |
| Body | Yellow rounded rect (40×20 px) |
| Arm | Shoulder → elbow line (4 px, gold) |
| Hand | Elbow → hand tip line (3 px, slightly darker) |
| Joints | Yellow circles at shoulder and elbow (4 px radius) |
| Hand tip | Red circle (5 px radius) |
| Wheels | Two dark circles at `robot_x ± 12` on the ground line |
| HUD | Step count, action, reward, position with delta (top-left, Courier 14pt bold) |

Each frame: pump events (ESC/close exits), one learning step,
redraw, `clock.tick(60)`, optional `time.sleep(delay)`.

---

## Pygame Dependency

Pygame is **optional**. Install it with:

```bash
pip install pygame
```

Without Pygame, everything runs in text mode automatically:

- `mars_rover.py` falls back to `TextMarsDisplay` (box-drawing ASCII)
- `autograder.py` and `check.py` **never** use the GUI — they
  suppress stdout during test execution
- `env/crawler.py`'s `run_crawler()` trains in text mode with
  progress prints

Both `gui_display.py` and `crawler_gui.py` import Pygame inside a
`try/except ImportError` block, so the modules load cleanly on
machines without Pygame. Constructing a `MarsDisplay` without
Pygame raises a clear `ImportError` pointing at the text-mode
alternative.

---

## Import Examples

```python
# Pygame GUI (requires pygame)
from display.gui_display import MarsDisplay

# Text fallback (no dependencies)
from display.text_display import TextMarsDisplay

# Crawler GUI (requires pygame)
from display.crawler_gui import run_crawler_gui
```

The typical pattern in `mars_rover.py`:

```python
try:
    from display.gui_display import MarsDisplay
    display = MarsDisplay(mdp)
except Exception:
    from display.text_display import TextMarsDisplay
    display = TextMarsDisplay(mdp)
```
