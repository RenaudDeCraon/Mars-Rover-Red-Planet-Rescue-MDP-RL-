"""Pygame crawler visualisation. 800x450, conditional import."""

import math, time
from typing import Any

try:
    import pygame
    _PG = True
except ImportError:
    pygame = None  # type: ignore[assignment]
    _PG = False


def run_crawler_gui(env: Any, agent: Any, steps: int = 500, delay: float = 0.0) -> None:
    """Train ``agent`` on ``env`` for ``steps`` with a Pygame window.

    ESC or window close exits early; ``delay`` sleeps between frames.
    Raises ImportError if pygame is missing.
    """
    if not _PG:
        raise ImportError(
            "run_crawler_gui requires pygame. Install with 'pip install pygame', "
            "or use env.crawler.run_crawler for the text-mode driver.")

    BG, GROUND, LINE = (25, 15, 12), (120, 70, 45), (100, 60, 35)
    BODY, ARM, HAND = (230, 200, 70), (180, 150, 80), (160, 130, 70)
    JOINT, TIP, WHEEL, HUD = (235, 215, 95), (220, 60, 60), (35, 25, 20), (240, 220, 180)

    pygame.init()
    screen = pygame.display.set_mode((800, 450))
    pygame.display.set_caption("Mars Crawler -- Q-learning")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Courier New', 14, bold=True)

    start_x = env.robot_x
    agent.start_episode()
    last_action, last_reward, running, step = '-', 0.0, True, 0

    while running and step < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        state = env.get_current_state()
        action = agent.get_action(state)
        if action is None:
            break
        next_state, reward = env.do_action(action)
        agent.observe_transition(state, action, next_state, reward)
        step += 1
        last_action, last_reward = action, reward

        # Background + ground.
        screen.fill(BG)
        gy = int(env.ground_y)
        pygame.draw.rect(screen, GROUND, pygame.Rect(0, gy, 800, 450 - gy))
        pygame.draw.line(screen, LINE, (0, gy), (800, gy), 2)

        rx = env.robot_x
        # Body rect.
        pygame.draw.rect(screen, BODY, pygame.Rect(int(rx - 20), 280, 40, 20), border_radius=4)

        # Kinematics (duplicated to avoid reaching into env privates).
        arm_angle = env.arm_buckets[env.arm_index]
        hand_angle = env.hand_buckets[env.hand_index]
        sx = rx + env.robot_width / 2.0
        sy = env.ground_y - env.robot_height
        ex = sx + env.arm_length * math.cos(arm_angle)
        ey = sy - env.arm_length * math.sin(arm_angle)
        tot = arm_angle + hand_angle
        hx = ex + env.hand_length * math.cos(tot)
        hy = ey - env.hand_length * math.sin(tot)

        pygame.draw.line(screen, ARM, (int(sx), int(sy)), (int(ex), int(ey)), 4)
        pygame.draw.line(screen, HAND, (int(ex), int(ey)), (int(hx), int(hy)), 3)
        pygame.draw.circle(screen, JOINT, (int(sx), int(sy)), 4)
        pygame.draw.circle(screen, JOINT, (int(ex), int(ey)), 4)
        pygame.draw.circle(screen, TIP, (int(hx), int(hy)), 5)
        pygame.draw.circle(screen, WHEEL, (int(rx - 12), gy), 6)
        pygame.draw.circle(screen, WHEEL, (int(rx + 12), gy), 6)

        # HUD.
        lines = [f'STEP    : {step}/{steps}',
                 f'ACTION  : {last_action}',
                 f'REWARD  : {last_reward:+.4f}',
                 f'POSITION: x = {env.robot_x:7.2f}  (delta = {env.robot_x - start_x:+.2f})']
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, HUD), (10, 10 + i * 18))

        pygame.display.flip()
        clock.tick(60)
        if delay > 0:
            time.sleep(delay)

    agent.stop_episode()
    pygame.quit()
