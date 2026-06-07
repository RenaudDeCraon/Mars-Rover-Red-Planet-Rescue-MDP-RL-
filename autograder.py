"""
autograder.py -- file-based Mars Rover autograder (Python 3.10+).

Walks ``test_cases/`` question directories, reads each CONFIG +
``.test`` + ``.solution`` triplet, instantiates the matching test
class from ``grading.mars_test_classes``, executes every test case
through a :class:`grading.grading.Grades` grade-book, and prints
the final results.

Usage::

    python autograder.py                       # run all questions
    python autograder.py -q q1                 # run one question
    python autograder.py -q q1 -q q3           # run a subset
    python autograder.py --test-directory DIR  # custom test root
"""

import argparse
import io
import os
import sys
from typing import Any

from grading.grading import Grades
from grading.test_parser import parse_test, parse_solution
from grading import test_classes as tc_mod
from grading import mars_test_classes as mtc


BANNER = (
    "\n"
    "    ======================================================\n"
    "       \U0001F52C  MARS ROVER AUTOGRADER  \u00b7  75 POINTS\n"
    "       CS 451/551 -- Introduction to AI\n"
    "       Mars Rover: Red Planet Rescue\n"
    "    ======================================================\n"
)

# Map the class name strings that appear in .test / CONFIG files to
# the actual Python classes so that the loader can instantiate them.
TEST_CLASS_MAP: dict[str, type] = {
    'ValueIterationTest':        mtc.ValueIterationTest,
    'GridPolicyTest':            mtc.GridPolicyTest,
    'QLearningTest':             mtc.QLearningTest,
    'EpsilonGreedyTest':         mtc.EpsilonGreedyTest,
}

QUESTION_CLASS_MAP: dict[str, type] = {
    'NumberPassedQuestion':  tc_mod.NumberPassedQuestion,
    'PassAllTestsQuestion':  tc_mod.PassAllTestsQuestion,
}


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------


def _load_question(q_dir: str, q_name: str):
    """Parse one question directory → (Question, [solution_dicts])."""
    config = parse_test(os.path.join(q_dir, 'CONFIG'))
    max_pts = int(config['max_points'])
    q_cls_name = config.get('class', 'NumberPassedQuestion')
    display = config.get('display_name', q_name)

    q_cls = QUESTION_CLASS_MAP[q_cls_name]
    question = q_cls(max_pts, display)

    test_files = sorted(f for f in os.listdir(q_dir) if f.endswith('.test'))
    sol_dicts: list[dict] = []

    for tf in test_files:
        td = parse_test(os.path.join(q_dir, tf))
        td['path'] = os.path.join(q_dir, tf)
        td.setdefault('name', tf.replace('.test', ''))

        tc_cls = TEST_CLASS_MAP[td['class']]
        question.add_test_case(tc_cls(question, td))

        sf = os.path.join(q_dir, tf.replace('.test', '.solution'))
        if os.path.isfile(sf) and os.path.getsize(sf) > 0:
            sol_dicts.append(parse_solution(sf))
        else:
            sol_dicts.append({})

    return question, sol_dicts


def _discover_questions(test_dir: str, selected: list[str] | None):
    """Return sorted list of question names (subdirs of test_dir)."""
    names = sorted(
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    )
    if selected:
        names = [n for n in names if n in set(selected)]
    return names


# ------------------------------------------------------------------
# Summary emoji
# ------------------------------------------------------------------


def _summary_line(pct: float) -> str:
    if pct >= 95:
        return '\U0001F3C6  Outstanding mission! Rover-7 returns with all samples.'
    if pct >= 80:
        return '\U0001F680  Great work -- Rover-7 is mission-ready.'
    if pct >= 60:
        return '\U0001F6F0  Good progress -- a few systems still need tuning.'
    if pct >= 40:
        return '\U0001F527  Keep working -- the rover needs more engineering time.'
    return '\U0001F4E1  Need more effort -- check the unfilled student stubs.'


# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------


def run_autograder(
    test_dir: str,
    selected: list[str] | None = None,
) -> Grades:
    """Walk *test_dir*, execute every question, return Grades."""
    q_names = _discover_questions(test_dir, selected)

    # Build rubric + load each question's .test/.solution data.
    rubric: list[tuple[str, int]] = []
    loaded: dict[str, Any] = {}
    for qn in q_names:
        qd = os.path.join(test_dir, qn)
        question, sol_dicts = _load_question(qd, qn)
        rubric.append((qn, question.max_points))
        loaded[qn] = (question, sol_dicts, qd)

    grades = Grades('Mars Rover: Red Planet Rescue', rubric)

    for qn in q_names:
        question, sol_dicts, qd = loaded[qn]
        grades.start_question(qn)

        # Suppress training banners from agents while grading.
        old_out = sys.stdout
        sys.stdout = io.StringIO()
        try:
            question.execute(grades, {}, sol_dicts)
        finally:
            sys.stdout = old_out

    return grades


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Mars Rover RL -- autograder (75 points)',
    )
    parser.add_argument(
        '-q', '--question',
        action='append', default=[],
        choices=['q1', 'q2', 'q3', 'q4', 'q5'],
        help='run a single question (repeatable). Default: run all.',
    )
    parser.add_argument(
        '--test-directory', default='test_cases',
        help='path to the test_cases/ root (default: test_cases)',
    )
    parser.add_argument(
        '--no-graphics', action='store_true',
        help='suppress any GUI windows (currently a no-op)',
    )
    args = parser.parse_args(argv)

    print(BANNER)

    grades = run_autograder(
        test_dir=args.test_directory,
        selected=args.question or None,
    )

    # Print results.
    earned, possible = grades.get_total()
    pct = 100.0 * earned / possible if possible else 0.0
    print()
    grades.produce_output()
    print(f'  {_summary_line(pct)}')
    print('=' * 56)


if __name__ == '__main__':
    main()
