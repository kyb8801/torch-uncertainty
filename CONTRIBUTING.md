# Contributing to TorchUncertainty

TorchUncertainty is in early development stage. We are looking for
contributors to help us build a comprehensive library for uncertainty
quantification in PyTorch.

We are particularly open to any comment that you would have on this project.
Specifically, we are open to changing these guidelines as the project evolves.

## The scope of TorchUncertainty

TorchUncertainty can host any method - if possible linked to a paper -
roughly contained in the following fields:

- Uncertainty quantification in general, including Bayesian deep learning,
Monte Carlo dropout, ensemble methods, etc.
- Out-of-distribution detection methods
- Applications (e.g. object detection, segmentation, etc.)

## Common guidelines

### Clean development install of TorchUncertainty

If you are interested in contributing to torch_uncertainty, we first advise you
to follow the following steps to reproduce a clean development environment
ensuring that continuous integration does not break.

1. Install `uv` following the steps from their [website](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the repository
3. Install torch-uncertainty with the dev packages:
   - `uv sync --extra gpu` for GPU-based systems
   - `uv sync --extra cpu` if no GPUs are available

   **Note:** failure to include the extra flag will result in the GPU version of PyTorch without CUDA and might cause issues.

4. Install pre-commit hooks with:

```sh
uv run pre-commit install
```

### Build the documentation locally

Navigate to `./docs` and build the documentation with:

```sh
uv run make html
```

Optionally, specify `html-noplot` instead of `html` to avoid running the tutorials.
This option is necessary if you only have a CPU on your machine.

### Guidelines

#### Commits

We use `ruff` for code formatting, linting, and imports (as a drop-in
replacement for `black`, `isort`, and `flake8`). The `pre-commit` hooks will
ensure that your code is properly formatted and linted before committing.

To ensure that your code complies with the standards, run the following and check for warnings:

```sh
uv run ruff check --fix
```

And then:

```sh
uv run ruff format
```

Please ensure that the tests are passing on your machine before pushing on a
PR. This will avoid multiplying the number of featureless commits. To do this,
run, at the root of the folder:

```sh
uv run pytest tests
```

##### Commit message convention

I (O.L.) follow a structured commit message format inspired by
[gitmoji](https://gitmoji.dev/) and the
[Conventional Commits](https://www.conventionalcommits.org/) specification:

```
:emoji:(scope): description
```

- **Emoji:** pick the most appropriate emoji from [gitmoji](https://gitmoji.dev/)
  to categorize the change at a glance.
- **Scope:** a short identifier in parentheses describing the area of the codebase
  affected (e.g. `packed`, `metrics`, `docs`, `CI`, `all`).
- **Description:** a concise summary of the change after the `:` separator.
- **Breaking changes:** use `!` before `:` to signal a breaking change.

Here are some examples from the project:

| Commit message | Meaning |
|---|---|
| `:bug:(cli): fix self.model not being a property` | Bug fix in the CLI module |
| `:sparkles:(losses): add the DER loss` | New feature in losses |
| `:recycle:(segformer): remove init redundancies` | Refactoring of segformer |
| `:rotating_light:(all): ruff format` | Fixing linter warnings across the codebase |
| `:memo:(tabular): add some documentation` | Documentation for the tabular module |
| `:white_check_mark:(coverage): fix coverage misses` | Test improvements |
| `:wrench:(uv): update uv & workflows` | Configuration/tooling changes |
| `:boom:(routines)!: rename eval to test` | Breaking change in routines |

Some of the most commonly used gitmoji in this project:

| Emoji | Code | Use |
|-------|------|-----|
| ✨ | `:sparkles:` | New feature |
| 🐛 | `:bug:` | Bug fix |
| ♻️ | `:recycle:` | Refactor code |
| 📝 | `:memo:` | Documentation |
| 🚨 | `:rotating_light:` | Fix linter warnings |
| ✅ | `:white_check_mark:` | Add/fix tests |
| 🔧 | `:wrench:` | Configuration files |
| 🔥 | `:fire:` | Remove code or files |
| ⬆️ | `:arrow_up:` | Upgrade dependencies |
| 💚 | `:green_heart:` | Fix CI build |
| 👷 | `:construction_worker:` | CI system changes |
| 🎨 | `:art:` | Improve code structure/format |
| 💥 | `:boom:` | Breaking changes |
| 👌 | `:ok_hand:` | Code review changes |

For the full list of available emojis, visit [gitmoji.dev](https://gitmoji.dev/).

You don't need to follow this convention.

#### Pull requests

To make your changes, create a branch on a personal fork and create a PR when your contribution
is mostly finished or if you need help.

##### PR naming convention

Pull requests should follow the same naming convention as commits:

```
:emoji:(scope): description
```

For instance: `:sparkles:(datamodules): add support for CIFAR-100-C`.

##### PR labels

Please flag your PRs with the appropriate labels to help maintainers triage and
track changes. Common labels include:

- `enhancement` — new feature or improvement
- `bug` — bug fix
- `documentation` — documentation changes
- `refactor` — code restructuring without behavior change
- `tests` — test additions or fixes
- `dependencies` — dependency updates
- `breaking-change` — introduces a breaking change
- `need-help` — you need assistance from a maintainer

##### PR checklist

Check that your PR complies with the following conditions:

- The name of your branch is not `main` nor `dev` (see issue [#58](https://github.com/torch-uncertainty/torch-uncertainty/issues/58))
- Your PR does not reduce the code coverage
- Your code is documented: the function signatures are typed, and the main functions have clear
docstrings
- Your code is mostly original, and the parts coming from licensed sources are explicitly
stated as such
- If you implement a method, please add a reference to the corresponding paper in the
["References" page](https://torch-uncertainty.github.io/references.html)
- Also, remember to add TorchUncertainty to the list of libraries implementing this reference
on [PapersWithCode](https://paperswithcode.com)

If you need help to implement a method, increase the coverage, or solve ruff-raised errors,
create the PR with the `need-help` flag and explain your problems in the comments. A maintainer
will do their best to help you.

### Datasets & Datamodules

We intend to include datamodules for the most popular datasets only.

### Post-processing methods

For now, we intend to follow scikit-learn style API for post-processing
methods (except that we use a validation dataset instead of a numpy array). You may get
inspiration from the already implemented
[temperature-scaling](https://github.com/torch-uncertainty/torch-uncertainty/blob/dev/torch_uncertainty/post_processing/calibration/temperature_scaler.py).

## License

If you feel that the current license is an obstacle to your contribution, let us
know, and we may reconsider. However, the models' weights hosted on Hugging
Face are likely to remain Apache 2.0.
