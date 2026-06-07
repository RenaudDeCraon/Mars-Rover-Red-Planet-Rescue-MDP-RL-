"""Pygame Mars grid display (3.10+).

Rock-textured terrain, glowing samples/craters, procedurally drawn
rover, particle system (dust/sparkle/lava), V/Q/policy overlays,
telemetry HUD. V/Q/P keys toggle overlay, arrows drive, ESC quits.
Pygame imported conditionally; missing pygame -> ImportError at
MarsDisplay construction.
"""

import math
import random
from typing import Any

try:
    import pygame
    _PG = True
except ImportError:
    pygame = None  # type: ignore[assignment]
    _PG = False

CELL_SIZE = 110
MARGIN_L, MARGIN_R, MARGIN_T = 50, 50, 80
HUD_H = 130

BG_TOP, BG_BOT = (15, 8, 20), (95, 45, 25)
TERRAIN = (106, 62, 40)
WALL_BASE, WALL_TOP, WALL_HI = (55, 32, 20), (125, 72, 38), (170, 110, 55)
SAMPLE_GLOW, SAMPLE_CRYSTAL = (90, 230, 140), (190, 255, 210)
CRATER_GLOW, CRATER_LAVA = (255, 95, 45), (255, 200, 80)
ROVER_BODY, ROVER_ACCENT = (200, 168, 64), (255, 215, 100)
PANEL_BLUE, ANTENNA_LIGHT = (70, 110, 200), (255, 80, 40)
CAMERA_IRIS, WHEEL = (140, 200, 255), (35, 25, 20)
HUD_BG, HUD_BORDER, HUD_TEXT = (22, 12, 10), (140, 80, 45), (240, 220, 180)
HUD_GOLD, HUD_GREEN, HUD_RED = (255, 220, 110), (90, 255, 130), (255, 90, 90)


class _Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'r', 'g', 'b', 'size')
    def __init__(self, x, y, vx, vy, life, r, g, b, size):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.r, self.g, self.b, self.size = r, g, b, size


class ParticleSystem:
    """Flat pool of particles with gravity and life-based alpha fade."""

    def __init__(self):
        self.particles: list[_Particle] = []

    def emit_dust(self, x, y, count=12):
        for _ in range(count):
            self.particles.append(_Particle(
                x + random.uniform(-8, 8), y + random.uniform(-4, 4),
                random.uniform(-0.8, 0.8), random.uniform(-1.8, -0.4),
                random.randint(18, 32),
                random.randint(140, 180), random.randint(90, 120),
                random.randint(55, 80), random.randint(2, 4)))

    def emit_sparkle(self, x, y, count=6):
        for _ in range(count):
            a = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.6, 1.4)
            self.particles.append(_Particle(
                x, y, math.cos(a) * s, math.sin(a) * s,
                random.randint(14, 24),
                random.randint(150, 220), 255, random.randint(180, 230),
                random.randint(2, 3)))

    def emit_lava(self, x, y, count=3):
        for _ in range(count):
            self.particles.append(_Particle(
                x + random.uniform(-6, 6), y,
                random.uniform(-0.3, 0.3), random.uniform(-1.2, -0.5),
                random.randint(20, 40),
                255, random.randint(120, 180), random.randint(30, 70),
                random.randint(2, 4)))

    def update_and_draw(self, surface):
        alive: list[_Particle] = []
        for p in self.particles:
            p.vy += 0.04
            p.x += p.vx; p.y += p.vy; p.life -= 1
            if p.life > 0:
                alpha = max(0, min(255, int(255 * p.life / p.max_life)))
                s = pygame.Surface((p.size * 2, p.size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (p.r, p.g, p.b, alpha), (p.size, p.size), p.size)
                surface.blit(s, (int(p.x) - p.size, int(p.y) - p.size))
                alive.append(p)
        self.particles = alive


class MarsDisplay:
    """Pygame display for a MarsGrid with value/Q/policy overlays."""

    def __init__(self, mdp: Any, mode: str = 'values'):
        if not _PG:
            raise ImportError(
                "MarsDisplay requires pygame. Install with 'pip install pygame', "
                "or use display.text_display.TextMarsDisplay for headless runs.")
        self.mdp = mdp
        self.mode = mode
        self.frame = 0
        self.should_quit = False
        self.window_w = MARGIN_L + mdp.width * CELL_SIZE + MARGIN_R
        self.window_h = MARGIN_T + mdp.height * CELL_SIZE + HUD_H

        pygame.init()
        pygame.display.set_caption("Mars Rover: Red Planet Rescue")
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont('Courier New', 22, bold=True)
        self.font_med = pygame.font.SysFont('Courier New', 14, bold=True)
        self.font_sml = pygame.font.SysFont('Courier New', 11)
        self.particles = ParticleSystem()

        rng = random.Random(42)
        self.stars = [(rng.randint(0, self.window_w), rng.randint(0, MARGIN_T),
                       rng.randint(80, 200)) for _ in range(100)]
        self.terrain_dots: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for x in range(mdp.width):
            for y in range(mdp.height):
                self.terrain_dots[(x, y)] = [
                    (rng.randint(4, CELL_SIZE - 4), rng.randint(4, CELL_SIZE - 4),
                     rng.randint(1, 2)) for _ in range(rng.randint(3, 8))]

    def _cell_rect(self, x, y):
        sy = MARGIN_T + (self.mdp.height - 1 - y) * CELL_SIZE
        return pygame.Rect(MARGIN_L + x * CELL_SIZE, sy, CELL_SIZE, CELL_SIZE)
    def _cell_center(self, x, y):
        r = self._cell_rect(x, y); return r.centerx, r.centery

    def draw(self, **kwargs: Any) -> None:
        self.frame += 1
        self.handle_events()
        if self.should_quit:
            return
        state = kwargs.get('state')
        if state is None:
            env = kwargs.get('env')
            if env is not None and hasattr(env, 'get_current_state'):
                state = env.get_current_state()
        agent = kwargs.get('agent')
        mode = kwargs.get('mode', self.mode)

        self._draw_sky()
        self._draw_stars()
        self._draw_title_bar()
        self._draw_grid(state, agent, mode)
        if state is not None and state != 'TERMINAL_STATE':
            self._draw_rover(state)
        self.particles.update_and_draw(self.screen)
        self._draw_hud(kwargs, mode)
        pygame.display.flip()
        self.clock.tick(60)

    def handle_events(self) -> str | None:
        """Pump events; return a movement action on arrow/space/enter."""
        action: str | None = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE: self.should_quit = True
                elif k == pygame.K_v: self.mode = 'values'
                elif k == pygame.K_q: self.mode = 'q_values'
                elif k == pygame.K_p: self.mode = 'policy'
                elif k == pygame.K_UP: action = 'north'
                elif k == pygame.K_DOWN: action = 'south'
                elif k == pygame.K_LEFT: action = 'west'
                elif k == pygame.K_RIGHT: action = 'east'
                elif k in (pygame.K_SPACE, pygame.K_RETURN): action = 'extract'
        return action

    def wait_for_key_action(self, legal_actions) -> str | None:
        while not self.should_quit:
            action = self.handle_events()
            if action is not None and action in legal_actions:
                return action
            self.clock.tick(60)
        return None

    def wait_for_key(self) -> None:
        while not self.should_quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.should_quit = True
                elif event.type == pygame.KEYDOWN:
                    return
            self.clock.tick(30)

    def close(self) -> None:
        pygame.quit()

    def _draw_sky(self):
        for y in range(self.window_h):
            t = min(1.0, y / self.window_h)
            r = int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * t)
            g = int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * t)
            b = int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * t)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.window_w, y))

    def _draw_stars(self):
        for sx, sy, base in self.stars:
            b = max(40, min(255, int(base + 40 * math.sin(self.frame * 0.05 + sx * 0.1))))
            self.screen.set_at((sx, sy), (b, b, b))

    def _draw_title_bar(self):
        t1 = self.font_big.render("MARS ROVER", True, HUD_GOLD)
        t2 = self.font_sml.render("RED PLANET RESCUE \u00b7 REINFORCEMENT LEARNING", True, HUD_TEXT)
        self.screen.blit(t1, (MARGIN_L, 10))
        self.screen.blit(t2, (MARGIN_L + t1.get_width() + 16, 20))
        pygame.draw.line(self.screen, HUD_BORDER, (MARGIN_L, MARGIN_T - 8),
                         (self.window_w - MARGIN_R, MARGIN_T - 8), 2)

    def _draw_grid(self, state, agent, mode):
        for x in range(self.mdp.width):
            for y in range(self.mdp.height):
                cell = self.mdp.cells[x][y]
                rect = self._cell_rect(x, y)
                if cell == '#':
                    self._draw_wall(rect)
                else:
                    self._draw_terrain(rect, x, y)
                    if isinstance(cell, float):
                        (self._draw_sample if cell > 0 else self._draw_crater)(rect, cell)
                    elif agent is not None and mode != 'grid':
                        self._draw_overlay(rect, (x, y), agent, mode)

    def _draw_terrain(self, rect, x, y):
        shade = 8 * math.sin((x * 13 + y * 7) * 0.1)
        base = (int(TERRAIN[0] + shade), int(TERRAIN[1] + shade), int(TERRAIN[2] + shade))
        pygame.draw.rect(self.screen, base, rect)
        dark = (base[0] - 15, base[1] - 15, base[2] - 10)
        for dx, dy, rr in self.terrain_dots[(x, y)]:
            pygame.draw.circle(self.screen, dark, (rect.x + dx, rect.y + dy), rr)
        pygame.draw.rect(self.screen, (base[0] - 20, base[1] - 20, base[2] - 12), rect, 1)

    def _draw_wall(self, rect):
        pygame.draw.rect(self.screen, WALL_BASE, rect)
        top = rect.inflate(-4, -4); top.y -= 3
        pygame.draw.rect(self.screen, WALL_TOP, top)
        pygame.draw.line(self.screen, WALL_HI, (top.x, top.y), (top.right, top.y), 2)
        for i in range(3):
            yy = top.y + 18 + i * 22
            pygame.draw.line(self.screen, WALL_BASE, (top.x + 6, yy), (top.right - 6, yy), 1)

    def _draw_sample(self, rect, reward):
        cx, cy = rect.centerx, rect.centery
        pulse = 0.5 + 0.5 * math.sin(self.frame * 0.08)
        gr = int(30 + pulse * 12)
        gs = pygame.Surface((gr * 2, gr * 2), pygame.SRCALPHA)
        pygame.draw.circle(gs, (*SAMPLE_GLOW, int(60 + 60 * pulse)), (gr, gr), gr)
        self.screen.blit(gs, (cx - gr, cy - gr))
        pts = [(cx, cy - 22), (cx + 16, cy), (cx, cy + 22), (cx - 16, cy)]
        pygame.draw.polygon(self.screen, SAMPLE_CRYSTAL, pts)
        pygame.draw.polygon(self.screen, SAMPLE_GLOW, pts, 2)
        pygame.draw.line(self.screen, (255, 255, 255), (cx - 5, cy - 6), (cx + 5, cy + 6), 2)
        lbl = self.font_med.render(f'+{reward:g}', True, (255, 255, 255))
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, rect.bottom - 20))
        if self.frame % 20 == 0: self.particles.emit_sparkle(cx, cy, 4)

    def _draw_crater(self, rect, reward):
        cx, cy = rect.centerx, rect.centery
        pulse = 0.5 + 0.5 * math.sin(self.frame * 0.07 + 1)
        gr = int(34 + pulse * 10)
        gs = pygame.Surface((gr * 2, gr * 2), pygame.SRCALPHA)
        pygame.draw.circle(gs, (*CRATER_GLOW, int(40 + 60 * pulse)), (gr, gr), gr)
        self.screen.blit(gs, (cx - gr, cy - gr))
        pygame.draw.circle(self.screen, (80, 35, 20), (cx, cy), 30)
        pygame.draw.circle(self.screen, (40, 18, 10), (cx, cy), 22)
        pygame.draw.circle(self.screen, CRATER_LAVA, (cx, cy), 10)
        pygame.draw.circle(self.screen, (255, 235, 140), (cx, cy), 4)
        lbl = self.font_med.render(f'{reward:g}', True, (255, 200, 180))
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, rect.bottom - 20))
        if self.frame % 30 == 0: self.particles.emit_lava(cx, cy, 3)

    def _draw_overlay(self, rect, state, agent, mode):
        if mode == 'values':
            try: v = float(agent.get_value(state))
            except Exception: return
            col = HUD_GREEN if v > 0 else HUD_RED if v < 0 else (130, 130, 130)
            surf = self.font_med.render(f'{v:+.2f}', True, col)
            self.screen.blit(surf, (rect.centerx - surf.get_width() // 2,
                                    rect.centery - surf.get_height() // 2))
        elif mode == 'q_values':
            try:
                actions = self.mdp.get_possible_actions(state)
                qs = {a: float(agent.get_q_value(state, a)) for a in actions}
            except Exception:
                return
            pygame.draw.line(self.screen, HUD_BORDER, rect.topleft, rect.bottomright, 1)
            pygame.draw.line(self.screen, HUD_BORDER, rect.topright, rect.bottomleft, 1)
            positions = {'north': (rect.centerx, rect.top + 14),
                         'south': (rect.centerx, rect.bottom - 14),
                         'east': (rect.right - 20, rect.centery),
                         'west': (rect.left + 20, rect.centery),
                         'extract': (rect.centerx, rect.centery)}
            for a, q in qs.items():
                col = HUD_GREEN if q > 0 else HUD_RED if q < 0 else (160, 160, 160)
                surf = self.font_sml.render(f'{q:+.1f}', True, col)
                px, py = positions.get(a, (rect.centerx, rect.centery))
                self.screen.blit(surf, (px - surf.get_width() // 2, py - surf.get_height() // 2))
        elif mode == 'policy':
            try: action = agent.get_policy(state)
            except Exception: return
            self._draw_policy_arrow(rect, action)

    def _draw_policy_arrow(self, rect, action):
        cx, cy = rect.centerx, rect.centery
        if action == 'extract':
            pygame.draw.circle(self.screen, HUD_GOLD, (cx, cy), 14, 3)
            return
        dx, dy = {'north': (0, -1), 'south': (0, 1),
                  'east': (1, 0), 'west': (-1, 0)}.get(action, (0, 0))
        if (dx, dy) == (0, 0): return
        length = 34
        tx, ty = cx + dx * length, cy + dy * length
        glow = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.line(glow, (*HUD_GOLD, 90),
                         (length, length), (length + dx * length, length + dy * length), 10)
        self.screen.blit(glow, (cx - length, cy - length))
        pygame.draw.line(self.screen, HUD_GOLD, (cx, cy), (tx, ty), 3)
        perp = (-dy * 8, dx * 8)
        back = (tx - dx * 10, ty - dy * 10)
        pts = [(tx, ty), (back[0] + perp[0], back[1] + perp[1]),
               (back[0] - perp[0], back[1] - perp[1])]
        pygame.draw.polygon(self.screen, HUD_GOLD, pts)

    def _draw_rover(self, state):
        if not isinstance(state, tuple) or len(state) != 2: return
        cx, cy = self._cell_center(*state)
        cy += int(math.sin(self.frame * 0.1) * 2)
        shadow = pygame.Surface((60, 16), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 100), (0, 0, 60, 16))
        self.screen.blit(shadow, (cx - 30, cy + 22))
        glow = pygame.Surface((80, 80), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*ROVER_ACCENT, 40), (40, 40), 36)
        self.screen.blit(glow, (cx - 40, cy - 40))
        body = pygame.Rect(cx - 24, cy - 12, 48, 24)
        pygame.draw.rect(self.screen, ROVER_BODY, body, border_radius=5)
        pygame.draw.rect(self.screen, ROVER_ACCENT, body, 2, border_radius=5)
        pygame.draw.rect(self.screen, ROVER_BODY, pygame.Rect(cx - 14, cy - 20, 28, 10), border_radius=3)
        for sx in (cx - 46, cx + 24):
            panel = pygame.Rect(sx, cy - 8, 22, 16)
            pygame.draw.rect(self.screen, PANEL_BLUE, panel)
            for i in range(1, 4):
                pygame.draw.line(self.screen, (30, 60, 140),
                                 (sx + i * 5, panel.top), (sx + i * 5, panel.bottom), 1)
            pygame.draw.line(self.screen, (140, 170, 220),
                             (sx + 22, cy), (cx - 24 if sx < cx else cx + 24, cy), 1)
        pygame.draw.line(self.screen, ROVER_ACCENT, (cx, cy - 20), (cx, cy - 30), 2)
        blink = ((self.frame // 20) % 2) == 0
        pygame.draw.circle(self.screen, ANTENNA_LIGHT if blink else (120, 50, 30), (cx, cy - 32), 3)
        pygame.draw.circle(self.screen, (20, 15, 10), (cx + 8, cy - 4), 4)
        pygame.draw.circle(self.screen, CAMERA_IRIS, (cx + 8, cy - 4), 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx + 9, cy - 5), 1)
        self.screen.blit(self.font_sml.render('7', True, (30, 20, 10)), (cx - 4, cy - 2))
        for wx in (cx - 16, cx + 16):
            pygame.draw.circle(self.screen, WHEEL, (wx, cy + 14), 6)
            ang = self.frame * 0.15
            for i in range(3):
                a = ang + i * (2 * math.pi / 3)
                pygame.draw.line(self.screen, (90, 60, 40), (wx, cy + 14),
                                 (wx + int(math.cos(a) * 5), cy + 14 + int(math.sin(a) * 5)), 1)
        if self.frame % 15 == 0: self.particles.emit_dust(cx, cy + 18, 4)

    def _draw_hud(self, kwargs, mode):
        panel = pygame.Rect(MARGIN_L, self.window_h - HUD_H - 10,
                            self.window_w - MARGIN_L - MARGIN_R, HUD_H)
        pygame.draw.rect(self.screen, HUD_BG, panel)
        pygame.draw.rect(self.screen, HUD_BORDER, panel, 2)
        for yy in range(panel.top + 2, panel.bottom - 2, 3):
            pygame.draw.line(self.screen, (35, 18, 14), (panel.left + 2, yy), (panel.right - 2, yy))
        pygame.draw.rect(self.screen, ROVER_BODY, (panel.x + 10, panel.y + 18, 16, 8))
        pygame.draw.circle(self.screen, WHEEL, (panel.x + 13, panel.y + 27), 2)
        pygame.draw.circle(self.screen, WHEEL, (panel.x + 23, panel.y + 27), 2)
        self.screen.blit(self.font_med.render('ROVER-7 TELEMETRY', True, HUD_GOLD),
                         (panel.x + 40, panel.y + 10))
        self.screen.blit(self.font_sml.render('MARS EXPLORATION AGENCY', True, HUD_TEXT),
                         (panel.x + 40, panel.y + 30))
        lines = [f'MODE    : {mode.upper()}',
                 f'POSITION: {kwargs.get("state", "-")}',
                 f'STEP    : {kwargs.get("step", "-")}',
                 f'ACTION  : {kwargs.get("action", "-")}']
        for i, line in enumerate(lines):
            self.screen.blit(self.font_sml.render(line, True, HUD_TEXT),
                             (panel.x + 40, panel.y + 50 + i * 14))
        reward = kwargs.get('reward')
        if reward is not None:
            col = HUD_GREEN if reward > 0 else HUD_RED if reward < 0 else HUD_TEXT
            self.screen.blit(self.font_sml.render(f'REWARD  : {reward:+.3f}', True, col),
                             (panel.x + 220, panel.y + 64))
        for i in range(5):
            h = 4 + i * 3 + int(3 * math.sin(self.frame * 0.2 + i))
            pygame.draw.rect(self.screen, HUD_GREEN,
                             (panel.right - 80 + i * 8, panel.y + 28 - h, 5, h))
        s = self.font_sml.render('V/Q/P: view  arrows: drive  ESC: quit', True, HUD_TEXT)
        self.screen.blit(s, (panel.right - s.get_width() - 10, panel.bottom - 16))
