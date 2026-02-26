# Contributing

We are actively developing PDX and accepting contributions! Any kind of PR is welcome. 

These are our current priorities:

**Features**:
- Inserts and Updates (wip).
- Out-of-core execution (disk-based setting).
- Implement multi-threading capabilities.
- Add PDX to the [VIBE benchmark](https://vector-index-bench.github.io/).
- Create a documentation.

**Improvements**:
- Regression tests on CI.


## Getting Started

1. **Fork the repository** on GitHub and create a feature branch:
```bash
git checkout -b my-feature
```

2. **Make your changes.**
3. **Run the test suite** locally before submitting your PR.
4. **Open a Pull Request (PR)** against the `main` branch.

> [!IMPORTANT]
> Let us know in advance if you plan implementing a big feature!

## Testing

All PRs must pass the full test suite in CI. Before submitting a PR, you should run tests locally:

```bash
# C++ tests
cmake . -DPDX_COMPILE_TESTS=ON
make -j$(nproc) tests
ctest .
```

Tests are also prone to bugs. If that is the case, please open an Issue.

## Submitting a PR

* Open your PR against the **`main` branch**.
* Make sure your branch is **rebased on top of `main`** before submission.
* Verify that **CI passes**.
* Keep PRs focused — small, logical changes are easier to review and merge.

## Coding Style
* Function, Class, and Struct names: `PascalCase`
* Variables and Class/Struct member names: `snake_case`
* Constants and magic variables: `UPPER_SNAKE_CASE`
* Avoid `new` and `delete`
* There is a `.clang-format` in the project. Make sure to adhere to it. We have provided scripts to check and format the files within the project:
```bash
pip install clang-format==18.1.8
./scripts/format_check.sh   # Checks the formatting
./scripts/format.sh         # Fix the formatting
```

## Communication

* Use GitHub Issues for bug reports and feature requests.
