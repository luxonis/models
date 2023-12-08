# Contributing to LuxonisTrain

This guide is intended for our internal development team.
It outlines our workflow and standards for contributing to this project.

## Table of Contents

- [Pre-commit Hook](#pre-commit-hook)
- [GitHub Actions](#github-actions)
- [Making and Reviewing Changes](#making-and-reviewing-changes)
- [Notes](#notes)

## Pre-commit Hook

We use a pre-commit hook to ensure code quality and consistency:

1. Install pre-commit (see [pre-commit.com](https://pre-commit.com/#install)).
2. Clone the repository and run `pre-commit install` in the root directory.
3. The pre-commit hook runs automatically on `git commit`.

## GitHub Actions

In addition to the pre-commit hook, our GitHub Actions workflow includes tests that must pass before merging:

1. Tests are run automatically when you open a pull request.
2. Review the GitHub Actions output if your PR fails.
3. Fix any issues to ensure that both the pre-commit hooks and tests pass.

## Making and Reviewing Changes

1. Make changes in a new branch.
2. Test your changes locally.
3. Commit (pre-commit hook will run).
4. Push to your branch and create a pull request. Always request a review from:
   - [Martin Kozlovský](https://github.com/kozlov721)
     - His permission is required for merge
   - [Matija Teršek](https://github.com/tersekmatija)
   - [Conor Simmons](https://github.com/conorsim)
5. The team will review and merge your PR.
