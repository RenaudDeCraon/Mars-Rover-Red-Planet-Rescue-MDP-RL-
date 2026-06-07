"""
project_params.py
-----------------
Project-wide metadata for the Mars Rover RL homework.

Used by the autograder and the main entry point to locate the
student files, the test-case classes, and to tag output with the
project name. Keeping these constants in one place lets us change
(for example) the canonical list of student files without having
to hunt through every other module.
"""

# Comma-separated list of the student-editable files, relative to
# the project root. Used by the autograder / dispatcher to decide
# which files count as "student code" when running tests and
# presenting error messages.
STUDENT_CODE_DEFAULT = (
    'students/analysis.py,'
    'students/qlearning_agents.py,'
    'students/value_iteration_agents.py'
)

# Dotted path to the module holding the project-specific
# TestCase subclasses. The grading framework imports this name
# dynamically so the same grading engine can be reused across
# multiple course projects.
PROJECT_TEST_CLASSES = 'grading.mars_test_classes'

# Human-readable name of the project, displayed in banners,
# reports, and GUI titles.
PROJECT_NAME = 'Mars Rover: Red Planet Rescue'
