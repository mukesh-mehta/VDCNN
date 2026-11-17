# Validation record

The maintained workflow was checked on Linux with Python 3.12.

- `python -m pytest -q`: **8 tests passed** locally.
- README command-line examples completed successfully using synthetic data.
- GitHub Actions runs the maintained test suite and command-line smoke checks.
- Historical notebooks, competition training, and leaderboard scores were not reproduced.

Root requirements describe the maintained environment. The historical environment,
where retained, is for reference and is not the setup used for these checks.
