# Git, GitHub and CI/CD learning plan

## What the terms mean

- **Git** records local versions of the project.
- **GitHub** hosts the Git repository and supports collaboration.
- **Continuous integration (CI)** automatically checks each proposed change.
- **Continuous delivery (CD)** prepares a validated artifact for release.
- **Continuous deployment** automatically releases a validated artifact to an
  environment. ForgeLM will add this only when a real server exists.

## First GitHub publication

Create an empty repository named `forgelm` on GitHub. Do not ask GitHub to add
a README, `.gitignore` or license because the local project already contains
its initial files. Then run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/forgelm.git
git push -u origin main
```

Alternatively, with an authenticated GitHub CLI:

```bash
gh repo create forgelm --public --source=. --remote=origin --push
```

## Workflow for every learning milestone

```text
issue -> milestone branch -> small commits -> local quality checks
      -> push -> pull request -> CI -> review -> merge -> tag
```

Even when working alone, use a pull request. It creates a permanent technical
record of the hypothesis, code, tests, measurements and limitations.

## CI progression

| Project stage | Automated checks added |
|---|---|
| 01 | Lint, type check, unit tests, integration smoke test |
| 02–06 | Mathematical reference tests and shape tests |
| 07 | Checkpoint-resume and deterministic-training smoke tests |
| 09 | Performance regression benchmarks with safe thresholds |
| 10 | Multi-process distributed smoke tests |
| 11–13 | Output-equivalence and inference benchmarks |
| 14 | Container build, security scan and staging deployment |
| 15–20 | Dataset validation and behavior/evaluation gates |
| 21–22 | Multimodal preprocessing and model-contract tests |

## CD progression

CD starts only after ForgeLM has an inference service:

1. build a versioned container for a Git tag;
2. run API and load-testing smoke tests;
3. publish the container to a registry;
4. deploy to a staging environment;
5. run health and evaluation gates;
6. require manual approval for the first production deployments;
7. add automated rollback based on health metrics.

Large training jobs must not run automatically on every commit. CI uses tiny
CPU fixtures, while expensive GPU experiments require an explicit manual
workflow and a recorded compute budget.

