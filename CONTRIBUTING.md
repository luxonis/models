# Contributing to LuxonisTrain

This guide is intended for our internal development team.
It outlines our workflow and standards for contributing to this project.

## Table of Contents

- [Pre-commit Hook](#pre-commit-hook)
- [GitHub Actions](#github-actions)
- [Making and Reviewing Changes](#making-and-reviewing-changes)
- [Notes](#notes)

## Installation

To install the package for development, first clone the repository:

```bash
git clone git@github.com:luxonis/luxonis-train.git
```

Then enter the cloned directory and install the package in editable mode with the `dev` extras:

```bash
pip install -e .[dev]
```
## CI/CD

We use GitHub Actions to run tests and enforce our coding style on each opened pull request. All the checks must pass in order for the PR to be merged.

### Style

We use a `pre-commit` hook to ensure code quality and consistency. The hook runs automatically on `git commit` and will fail if the code does not pass our style checks.

To configure the hook:
1. The `pre-commit` package is installed as part of the `dev` extras. You can also
install it manually with:
```bash
pip install pre-commit

```
1. Run `pre-commit install` in the root directory of the repository.
1. Now the hook will run automatically on `git commit`. The hook can modify files
by applying formatting and linting fixes.

### Tests

The tests can be run locally with:
```bash
pytest tests
```

GitHub Actions will run the tests automatically when a new PR is opened. Upon completion, the test reports will be posted in the PR comments.

## Making and Reviewing Changes

1. Make changes in a new branch.
1. Test your changes locally.
1. Commit (pre-commit hook will run).
1. Push to your branch and create a pull request. Always request a review from:
   - [Martin Kozlovský](https://github.com/kozlov721)
     - His permission is required for merge
   - [Matija Teršek](https://github.com/tersekmatija)
   - [Conor Simmons](https://github.com/conorsim)
1. The team will review and merge your PR.
