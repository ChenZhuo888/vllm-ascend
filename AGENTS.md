# vLLM Ascend Development Guidelines

This document provides instructions for contributors to the vLLM Ascend project. Please read and follow these guidelines to ensure code quality, maintainability, and consistency.

---

## Table of Contents

- [Setup and Environment](#setup-and-environment)
    - [Environment Variables](#environment-variables)
    - [Environment Variable Review Requirement](#environment-variable-review-requirement)
- [Testing](#testing)
    - [Unit and System Tests](#unit-and-system-tests)
    - [Running Tests](#running-tests)
- [Code Style](#code-style)
    - [Python Conventions](#python-conventions)
    - [Naming Conventions](#naming-conventions)
- [NPU-Specific Considerations](#npu-specific-considerations)
    - [Tensor item() Operations](#tensor-item-operations)
    - [Memory and Performance](#memory-and-performance)
- [Model and Plugin Architecture](#model-and-plugin-architecture)
    - [vLLM Ascend Plugin Architecture](#vllm-ascend-plugin-architecture)
    - [Patching Requirement](#patching-requirement)
    - [Model Runner Changes](#model-runner-changes)
- [Commit Messages and Pull Requests](#commit-messages-and-pull-requests)
    - [Commit Message Format](#commit-message-format)
- [Review Checklist](#review-checklist)
    - [Code Quality](#code-quality)
    - [Testing](#testing-1)
    - [Documentation](#documentation)
    - [NPU Considerations](#npu-considerations)
    - [Commit and PR](#commit-and-pr)
- [Quick Start for Contributors](#quick-start-for-contributors)
- [References](#references)

---

## Setup and Environment

### Environment Variables

All environment variables must be defined in `vllm_ascend/envs.py` using the centralized `env_variables` dictionary.

**Requirements:**

- Add documentation for each environment variable in the `env_variables` dict comment
- Specify default values and valid ranges
- Indicate whether the variable is sensitive (credentials, keys)

**Example:**

```python
import os

env_variables = {
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # ...
}
```

**Never**: Hardcode environment variable names throughout the codebase. Reference them from the central module using `from vllm_ascend import envs`.

### Environment Variable Review Requirement

**Strict Review Required**: All new environment variables must undergo code review.

Reviewers must verify:

- The variable name follows the `VLLM_ASCEND_*` naming convention
- Default value is appropriate for all supported hardware
- Documentation is added to the `env_variables` dict
- The variable is used in a performance-critical path

---

## Testing

### Unit and System Tests

**Requirement**: All new functionality requires corresponding tests.

- **Unit Tests (UT)**: Located in `tests/ut/`, cover core logic, edge cases, and error conditions
- **System Tests (ST)**: Located in `tests/e2e/`, verify end-to-end behavior and integration points
- **Nightly Tests**: Include benchmarks for NPU-specific code paths in `tests/e2e/nightly/`

**Test Coverage Guidelines:**

- New features: Tests must cover happy path and failure modes
- Bug fixes: Tests must include a regression test for the bug
- Performance-critical code: Include benchmarks and performance regression tests

### Running Tests

```bash


# Run specific unit test file
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run specific unit test
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# Run NPU-specific tests (requires NPU hardware)
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_aclgraph_accuracy.py::test_default_full_and_piecewise_res_consistency
```

**Requirement**: Run all tests locally before requesting review. Verify tests pass on NPU hardware for NPU-specific changes.

---

## Code Style

### Python Conventions

- **Imports**: All imports at the top of the file. Valid exceptions:
    - Circular imports (use inline imports)
    - Lazy loading for worker/isolation processes
    - Type-checking imports wrapped in `if TYPE_CHECKING:`

- **Global Variables**: Avoid new global variables. Pass dependencies explicitly through function parameters.

    **Allowed:**
    - Constants named `ALL_UPPER_CASE` (e.g., `MAX_BATCH_SIZE` in `envs.py`)
    - Immutable configuration objects

    **Requires Approval:**
    - Any new mutable global state

- **No Magic Numbers**: Use named constants with descriptive names:

    ```python
    # Bad
    if seq_len > 2048: ...

    # Good
    MAX_CONTEXT_LENGTH = 2048
    if seq_len > MAX_CONTEXT_LENGTH: ...
    ```

- **Descriptive Naming**: Use names that describe functionality, not implementation details.

    ```python
    # Bad
    is_deepseek_v3_r1
    flag1
    tmp_var

    # Good
    supports_dynamic_temperature
    uses_speculative_decoding
    ```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `NPUModelRunner`, `AscendSampler`, `ACLGraphManager`)
- **Functions/Methods**: `snake_case` (e.g., `forward_pass`, `compute_attention`)
- **Constants**: `ALL_UPPER_CASE` (e.g., `MAX_BATCH_SIZE`, `VLLM_ASCEND_ENABLE_NZ`)
- **Variables**: `snake_case` (e.g., `token_ids`, `sequence_lengths`)

---

## NPU-Specific Considerations

### Tensor item() Operations

**Warning**: `tensor.item()` operations cause synchronization overhead on NPU when the `tensor` is on device.

If the `tensor` is a device tensor, calling `item()` will triggers a synchronous data transfer from NPU to CPU. This can severely degrade performance in hot paths, causing `AsyncScheduler` to block here.

**Review Requirements:**

1. Profile performance impact before merging
2. Consider alternative patterns:
    - Keep values on device when possible
    - Batch operations to reduce sync frequency
    - Use device-side operations (e.g., `torch.argmax`, `torch.sum`)
3. Document when `item()` is unavoidable (e.g., logging, conditional logic)

**Example Patterns:**

```python
# Bad: In hot loop - causes sync per iteration
for tensor in tensors:
    value = tensor.item()

# Better: Batch operations - single sync
values = [t.item() for t in tensors]  # Single batch sync

# Good: Keep on device when possible
max_value = torch.max(tensor)  # No sync needed
if max_value > threshold:  # Comparison can stay on device
    ...
```

### Memory and Performance

Additional NPU-specific best practices:

- Avoid CPU-NPU memory transfers in hot paths
- Prefer in-place operations where safe (e.g., `x.add_()`, `x.mul_()`)
- Monitor memory fragmentation, especially for long-running processes
- Test with realistic workloads on actual NPU hardware (Ascend 910B/C)

---

## Model and Plugin Architecture

### vLLM Ascend Plugin Architecture

vLLM Ascend is a **hardware plugin** that integrates with upstream vLLM via the pluggable hardware interface. It does not add new model files directly.

**Required Pattern**: Model-specific functionality should be implemented via:

1. **Patching** (in `vllm_ascend/patch/`):
    - `vllm_ascend/patch/platform/` - Platform-level patches (distributed, scheduling)
    - `vllm_ascend/patch/worker/` - Worker-level patches (model-specific behavior)
    - Example: `patch_deepseek.py` modifies upstream Deepseek model behavior
    - Patch is not the best solution for all cases. Use it when necessary.

2. **Inheritance**:
    - `NPUModelRunner(GPUModelRunner)` - Extend vLLM model runner with NPU-specific behavior
    - `AscendSampler` - Extend vLLM sampler with NPU-specific operations
    - Add NPU-specific components via composition (e.g., `AclGraphManager`)
    - Custom Operators - NPU-specific custom operators (e.g., `AscendRMSNorm`)

3. **External upstream contributions** where appropriate

### Patching Requirement

**Strict Review Required**: All new patches must undergo thorough architectural review.

Reviewers must verify:

- The patch targets the correct upstream component
- The patch is minimal and focused
- Performance implications are understood
- A long-term plan exists for upstream contribution

**Example Patch Pattern:**

```python
# vllm_ascend/patch/worker/patch_deepseek.py
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

def forward(self, input_ids, positions, ...):
    # NPU-specific forward implementation
    ...

DeepseekV2Model.forward = forward  # Patch upstream class
```

### Model Runner Changes

**Strict Review Required**: All new behaviors added to `model_runner` must undergo thorough architectural review.

Reviewers must verify:

- The necessity of the new behavior (why can't this be in a patch?)
- Performance implications on NPU hardware
- Compatibility with existing model implementations
- Long-term maintainability and test coverage

**NPU Model Runner Files:**

- `vllm_ascend/worker/model_runner_v1.py` - vLLM v1 model runner
- `vllm_ascend/worker/v2/model_runner.py` - vLLM v2 model runner
- `vllm_ascend/_310p/model_runner_310p.py` - Ascend 310P model runner

---

## Commit Messages and Pull Requests

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format and **must include a sign-off**:

```bash
git commit -s -m "<type>: <summary>" -m "<body - explaining what changed and why>"
```

Or using the full message format:

```txt
<type>: <summary>

<body - explaining what changed and why>

Signed-off-by: Your Name <your.email@example.com>
```

**Valid Types**: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`

**Good Examples:**

```txt
feat(npu): add flash attention support for Ascend CANN

- Implements FlashAttention-2 kernel for NPU backend
- Reduces memory usage by 30% compared to baseline

fix(model_runner): correct padding token handling

- Fixes token padding that caused incorrect attention masks
- Addresses issue #1234

perf: avoid CPU-NPU sync in attention computation

- Inline computation to avoid tensor.item() calls
- Improves throughput by 15%
```

**Bad Examples:**

```txt
fix bug
add feature
update code
```

### Pull Request Title Format

PR titles should follow the format: `[Type][Module] Description`

- **Type**: The type of change (e.g., `CI`, `Doc`, `BugFix`, `Feat`, `Platform`, `Refactor`)
- **Module**: The affected module (optional, e.g., `Misc`, `Model`, `Worker`)
- **Description**: Brief description of the change

**Examples:**

- `[Doc][Misc] Update contribution guidelines`
- `[BugFix] Fix CPU binding logic`
- `[CI] Update image build workflow`

### Pull Request Template

When creating a PR, please follow the template in `.github/PULL_REQUEST_TEMPLATE.md` and ensure the following sections are completed:

> **Note**: The PR description will be automatically updated by GitHub Actions to include vLLM version info at the bottom. If you update the PR description via API or CLI, make sure to preserve the `- vLLM version:` and `- vLLM main:` lines.

- **What this PR does / why we need it?** - Clearly describe the changes and their purpose
- **Does this PR introduce _any_ user-facing change?** - Indicate if there are any user-visible changes
- **How was this patch tested?** - Describe how you tested the changes. Examples:
    - Unit tests added/updated: list the test files
    - Manual testing: provide the test steps and commands
    - CI testing: indicate if only CI verification is needed

---

## Review Checklist

Before merging, verify:

### Code Quality

- [ ] Code follows style guidelines (naming, imports, no magic numbers)
- [ ] No global state added without justification
- [ ] Patching pattern used correctly (if applicable)
- [ ] No direct model file additions

### Testing

- [ ] New tests added for new functionality (`tests/ut/` or `tests/e2e/`)
- [ ] Existing tests pass
- [ ] NPU-specific tests verified on actual hardware
- [ ] Performance benchmarks included where applicable

### Documentation

- [ ] Environment variables documented
- [ ] Public APIs documented
- [ ] User-facing changes reflected in docs

### NPU Considerations

- [ ] `tensor.item()` usage reviewed for performance impact
- [ ] No unnecessary CPU-NPU transfers in hot paths
- [ ] Memory usage verified on NPU hardware

### Commit and PR

- [ ] Commit messages are clear and descriptive, following Conventional Commits format
- [ ] **All commits are signed off** (`git commit -s`)
- [ ] PR is created from your fork repository, not directly from the main repository
- [ ] PR description is complete, following the PR template
- [ ] All review comments addressed

---

## Quick Start for Contributors

1. Install development dependencies: `pip install -e .[dev]`
2. Run tests: `pytest tests/`
3. Check linting: `ruff check vllm_ascend/`
4. Format code: `ruff format vllm_ascend/`
5. Make your changes following guidelines in this document
6. Add tests for new behavior
7. Run full test suite before committing
8. Commit with sign-off: `git commit -s`
9. Run linting check before pushing:
   ```bash
   bash format.sh ci
   ```
   > **Note**: This check is required for **all file types**, including markdown files. If `markdownlint` modifies files, re-add them with `git add` and commit again.
10. Push to your fork repository (NOT the main repository):

   ```bash
   git remote add myfork https://github.com/YOUR_USERNAME/vllm-ascend.git
   git push -u myfork your-branch-name
   ```

11. Create a PR from your fork to the main repository with clear description

---

## References

- [vLLM Hardware Plugin RFC](https://github.com/vllm-project/vllm/issues/11162)
- [Documentation](https://docs.vllm.ai/projects/ascend/en/latest/)
- [Contributors Guide](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html)

---

## AscendStore MP Engineering Principles

These rules apply to changes under
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp`.

### Priorities

1. Correctness and behavioral compatibility come first.
2. Code is written primarily for human maintainers. Prefer clear,
   unsurprising code over clever implementations.
3. Preserve existing in-process AscendStore behavior. MP implementations
   should adapt process-boundary differences through inheritance, composition,
   or adapters whenever practical.
4. Implement current requirements only. Do not introduce abstractions or
   features solely for hypothetical future needs.

### Architecture Boundaries

- RPC infrastructure owns transport, framing, routing, deadlines, execution,
  server health, shutdown, and transport-level errors. It must not understand
  KVCache business semantics.
- KVCache components own service registration, visibility, lifecycle
  coordination, degradation policy, and Lookup, Store, and Retrieve
  orchestration.
- Scheduler and Worker implementations should reuse the original
  `KVPoolScheduler` and `KVPoolWorker` semantics as much as possible.
- Backend adapters own Mooncake, Memcache, YuanRong, NPU IPC, and other
  backend-specific resource behavior.
- The vLLM connector translates between vLLM callbacks and KVCache operations.
  It must not absorb server-side orchestration or backend implementation
  details.
- Dependencies must point inward toward stable contracts. Business changes
  should not require modifying RPC infrastructure.

### Human-Readable Code

- A main method should read like a coherent narrative and expose the complete
  control flow.
- Prefer a wide-tree call structure: orchestration returns to a clear root
  method after each meaningful operation.
- Avoid deep chains of one-line forwarding methods. A helper is justified only
  when it owns a meaningful concern, invariant, reusable operation, or
  complexity.
- Each mutable state, lifecycle transition, retry policy, and failure policy
  must have one clear owner.
- Use precise domain names. Avoid generic names such as `Manager`, `Service`,
  `Context`, or `Handler` unless the object genuinely owns that complete
  responsibility.
- Comments should explain why a decision, invariant, ordering constraint, or
  workaround exists. Do not restate what the code already says.
- Keep formatting compact and compatible with PyCharm and Ruff. Avoid
  arbitrary or poetic line wrapping.

### Textual Organization

- The purpose of textual organization is to reveal the real architecture,
  control flow, ownership, and lifecycle of the code. It is not a comment
  coverage exercise and must not create a superficial taxonomy of every
  method.
- Establish the actual call chain and responsibility boundaries before
  reordering methods or adding sections. When the structure is unclear, do not
  use comments to make it appear settled; preserve the ambiguity as an
  architecture question to investigate.
- A class-level explanation belongs in the class docstring. Not every class or
  method needs a docstring, but explanatory prose immediately following a
  class declaration must not be written as a substitute block of comments.
- Constructors, properties, and nearby construction helpers normally read as
  one natural opening and do not need headings such as `Construction`,
  `State`, or `Helpers`.
- Use section banners only for substantial, coherent functional regions or
  execution phases whose boundary materially improves navigation. A class may
  begin with a section only when it is the first of several genuine peer
  regions, never merely to label the first methods.
- Use this banner form at the indentation level of the region, with real blank
  lines rather than comment-only spacer lines:

  ```python
  # ==============================
  # Scheduler-side methods
  # ==============================
  ```

- A section title should reveal more than the names of the methods below it.
  It should help a reader reconstruct a responsibility, pipeline stage,
  execution owner, or lifecycle transition while skimming the file. A
  one-method section is justified only when that method is itself a major
  transaction or lifecycle boundary.
- Explanatory text below a section banner is optional. Add it only to capture
  a non-obvious reason, invariant, ordering requirement, ownership rule, or
  compatibility constraint.
- Keep methods that participate in one operation close together. Prefer an
  order that exposes the public orchestration first and places its meaningful
  helpers nearby; use lifecycle order where it is the clearest narrative.
  Do not reorder methods merely for symmetry, alphabetic grouping, or route
  name consistency.
- Before applying textual reorganization across several files, propose the
  intended file order and section map. Establish the style on one
  representative file and review it before propagating the approach.

### Design Discipline

- Apply DDD lightly: use domain language and explicit boundaries, but do not
  mechanically introduce entities, repositories, factories, or domain
  services.
- Apply SOLID as judgment, not ceremony:
  - A component should have one reason to change, not necessarily one method.
  - Introduce an abstraction only when a real boundary or variation exists.
  - Keep interfaces narrow and shaped around their consumers.
  - Isolate external SDKs and unstable dependencies behind adapters.
- Small, obvious duplication is sometimes preferable to another layer of
  indirection.
- Do not create a class merely to hold functions or rename calls.
- Locks, conditions, events, queues, states, and background threads require a
  concrete concurrency responsibility. Do not add them defensively without an
  identified race or ownership requirement.

### Refactoring Rules

- Before changing architecture, identify the current call chain, state owner,
  invariants, failure behavior, concurrency model, and actual extension points.
- Separate structural refactoring from behavioral changes whenever possible.
- A structural refactor must preserve protocol layout, serialization behavior,
  task granularity, threading, queues, locks, resource ownership, failure
  semantics, and hot-path behavior unless the change is explicitly agreed
  upon.
- Make changes incrementally and keep every intermediate state understandable
  and reviewable.
- Update affected tests in the same change. Do not leave temporary handlers,
  compatibility branches, or test-only behavior behind unintentionally.
- Do not modify unrelated files or broaden the change boundary without
  explaining why it is necessary.
- Do not commit automatically unless explicitly requested. After each
  completed change, provide a Conventional Commit message with a clear summary
  and rationale.

### Verification

- Unit tests should cover behavior, boundaries, and important failure modes
  rather than internal implementation details.
- Real NPU and backend behavior must ultimately be verified by E2E tests.
  Mooncake is the primary functional baseline; backend-specific differences
  require their own targeted validation.
- Preserve useful error context across process boundaries. Runtime behavior
  and degradation policy must remain compatible with the in-process
  implementation while failures remain traceable.
- Performance optimization follows functional and architectural correctness,
  but refactoring must not introduce avoidable serialization, synchronization,
  copying, or scheduling overhead.
- If the required environment is unavailable, state which verification was not
  run instead of claiming success.
