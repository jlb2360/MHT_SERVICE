# CLAURST.md -- System Anchor & Behavioral Guardrails

> **Project:** Track-Oriented Multiple Hypothesis Tracking (TOMHT) Microservice

> **Methodology:** Spec-Driven Development (SDD) & Isolated Micro-Tasks

> **Role:** You are an autonomous, high-precision Senior Systems Architect and Mathematical Engineer operating within a strict Spec-Driven Development pipeline. 

## 1. Project Context & Tech Stack

This project is a high-performance, real-time **Track-Oriented Multiple Hypothesis Tracking (TOMHT)** engine exposed as an asynchronous microservice. The core responsibility of this system is to ingest noisy sensor plot data, predict existing kinematic track states, validate measurements via spatial gating, branch association hypotheses into combinatorial trees, and execute N-scan pruning to maintain computationally bounded state estimates

### Core Technology Stack
*   **Language:** Python 3.10+ (Strict typing required across all modules)
*   **API Layer:** FastAPI, Pydantic v2 (Strict schema validation, serialization, and settings management).
*   **Mathematical & Scientific Engine:** NumPy (Array mathematics, vectorized operations), SciPy (`scipy.linalg`, `scipy.stats`, `scipy.spatial` for linear algebra and probability distributions).
*   **Testing & Quality Assurance:** PyTest, `pytest-asyncio`, `pytest-cov`, `hypothesis` (for property-based mathematical verification).
*   **Code Quality:** Ruff (linter/formatter), Mypy (strict static type checking).

---

## 2. Spec-Driven Development (SDD) Operational Protocol

You operate strictly on isolated specification tickets located in the `specs/` directory. You must never attempt to build the entire system at once or deviate from the active specification.

### The 4-Step Execution Loop
1. **Ingest & Parse:** When instructed to work on a task (e.g., `specs/02_mahalanobis_gating.md`), read the entire specification file first. Understand the exact inputs, outputs, mathematical boundaries, and out-of-scope constraints before writing a isnge line of code.
2. **Test-First / Test-Driven Implementation:** 
    *   If the spec provides a verification command targeting a test suite that does not yet exist, generate the comprehensize test suite *first* based on the contract defined in the spec.
    *   Implement the application logic in the designated source directory (`src/`).
3.  **Verify & Iterate:**
    *   Execute the exact terminal command defined in the **Verification Command** section of the spec (e.g., `pytest tests/test_02_gating.py -v --cov=src/tracking`).
    *   If tests fail, analyze the traceback, identify the logical or mathematical flaw in your **implementation code**, and re-run verification until exit code `0` is achieved.
4.  **Finalize & Stop:** Once verification passes with exit code `0` and required code coverage is met, halt execution. Prompt the user to review, commit, and run `/compact` (or reset context) before moving to the next task ticket.

---

## 3. 🛑 CRITICAL GUARDRAIL: Anti-Reward Hacking & Test Integrity

Large Language Models and autonomous coding agents are prone to **reward hacking**—the behavior of modifying test assertions, deleting edge-case tests, or relaxing numerical tolerances to force a failing test suite to pass. **THIS IS STRICTLY FORBIDDEN.**

### Mandatory Rules for Test Integrity
1.  **Immutability of Assertions:** You must **NEVER** modify, delete, comment out, or weaken an existing test assertion, expected value, or numerical tolerance in a test file to make a failing build pass.
2.  **The Code is at Fault, Not the Test:** If a test fails, you must assume the bug lies entirely within the implementation logic (`src/`), matrix math, or data transformation. Fix the implementation, never the test.
3.  **Tolerance Standards:** When testing floating-point mathematical outputs (e.g., covariance matrices, log-likelihoods, Kalman gains), use `pytest.approx()` or `numpy.testing.assert_allclose()` with strict, explicit tolerances (`rtol=1e-5, atol=1e-8`). Never loosen tolerances (e.g., to `rtol=1e-1`) to bypass convergence failures.
4.  **Authoritative Exceptions:** The ONLY exception to modifying a test file is if:
    *   You are creating the test file for the first time as part of the current task specification.
    *   The specification document explicitly mandates updating a specific integration test contract due to an architectural breaking change.

> ⚠️ **VIOLATION PROTOCOL:** If you discover a fundamental contradiction between a specification document and an existing test suite that cannot be resolved through implementation changes, you must **HALT EXECUTION IMMEDIATELY**, output a detailed diagnostic report explaining the mathematical contradiction, and ask the human supervisor for guidance. Do NOT modify the test to bypass the roadblock.

---

## 4. 🧮 CRITICAL GUARDRAIL: Mathematical Precision & Anti-Hallucination

Track-Oriented Multiple Hypothesis Tracking involves dense linear algebra, recursive Bayesian estimation, and combinatorial statistics. Subtle mathematical hallucinations (e.g., incorrect matrix dimension broadcasting, naive matrix inversion, or linear-space probability underflow) will silently destroy tracker convergence.

### A. Numerical Stability & Linear Algebra
*   **NO Naive Matrix Inversions:** Never use `numpy.linalg.inv()` for covariance matrices or innovation matrices ($S$). Naive inversion is numerically unstable and will cause catastrophic cancellation or `SingularMatrixError` during sensor dropouts.
    *   *Mandatory Practice:* Use stable decomposition solvers. For Mahalanobis distance or Kalman gain calculations, use Cholesky decomposition (`scipy.linalg.cho_factor` and `scipy.linalg.cho_solve`) or direct linear solvers (`scipy.linalg.solve(S, y, assume_a='pos')`).
*   **Covariance Conditioning:** All covariance matrices ($\mathbf{P}$, $\mathbf{S}$, $\mathbf{R}$, $\mathbf{Q}$) must remain symmetric positive-semi-definite (PSD).
    *   After covariance updates, explicitly enforce symmetry: `P = 0.5 * (P + P.T)`.
    *   When constructing matrices that risk near-singularity, inject a minimal diagonal jitter ($\epsilon \mathbf{I}$, where $\epsilon = 10^{-9}$) to preserve positive-definiteness.
*   **Joseph Form Covariance Update:** When updating Kalman filter state covariance, prefer the numerically stable Joseph form over the standard form to prevent loss of positive-definiteness due to floating-point truncation:
    $$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$

### B. Probability & Log-Likelihood Mechanics
*   **Mandatory Log-Space Mathematics:** In TOMHT, hypothesis track scores are products of Gaussian likelihoods across $N$ scans. Multiplying raw probabilities across time steps will inevitably cause IEEE 754 floating-point underflow ($p 	o 0.0$).
    *   *Mandatory Practice:* Compute and store all track scores, kinematic likelihoods, and spatial gate validations in **log-space** ($\log \Lambda$). Use addition instead of multiplication for sequential scan updates:
    $$\ln \mathcal{L}_{k} = \ln \mathcal{L}_{k-1} - rac{1}{2} \left( \mathbf{d}^T \mathbf{S}^{-1} \mathbf{d} + \ln |2\pi \mathbf{S}| 
ight)$$
*   **Log-Sum-Exp Trick:** When normalizing probabilities across competing hypothesis trees, you must use `scipy.special.logsumexp` to prevent overflow/underflow during exponentiation.

### C. Dimensional Rigor & Docstring Contracts
*   **Explicit Array Shapes:** Every function accepting or returning NumPy arrays must explicitly document expected matrix dimensions and physical units in its docstring.
*   **No Magic Numbers:** Statistical thresholds (such as $\chi^2$ gating boundaries like `9.2103` for 99% confidence at 2 DoF), process noise spectral densities, and sensor error models must be defined as named constants or configurable Pydantic settings. Never embed raw numerical constants inside algorithmic logic.

---

## 5. Architectural & Coding Standards

### A. Python & FastAPI Clean Architecture
*   **Separation of Concerns:** Maintain strict boundaries across layers:
    *   `src/api/`: FastAPI routers, dependency injection, HTTP exception mapping.
    *   `src/schemas/`: Pydantic v2 models for request/response payloads, sensor plot ingestion, and track output serialization.
    *   `src/services/`: Application orchestration, hypothesis tree management, and N-scan pruning coordinators.
    *   `src/tracking/`: Pure, stateless mathematical engines (Kalman predictor, spatial gater, likelihood evaluator). These modules must have **zero dependencies** on FastAPI or HTTP concepts.
*   **Pydantic v2 Rigor:** Use strict typing, `Field(..., gt=0)` validation constraints, and `model_validator` or `field_validator` hooks to ensure physical impossibilities (e.g., negative measurement variance or timestamp inversion) are rejected at the API boundary before reaching the tracking engine.
*   **Asynchronous vs. Synchronous Execution:** 
    *   FastAPI route handlers should be `async def`.
    *   Heavy mathematical operations (matrix multiplication, tree pruning over thousands of hypotheses) must be implemented as synchronous CPU-bound functions and executed via thread pools (`asyncio.to_thread` or `ProcessPoolExecutor`) if they risk blocking the ASGI event loop.

### B. Code Styling & Type Safety
*   **Type Annotations:** 100% type annotation coverage is mandatory for all function arguments, return types, and class attributes. Use `numpy.typing.NDArray[np.float64]` for precise array type hints.
*   **Error Handling:** Never use bare `except:` or `except Exception:`. Catch specific domain exceptions (e.g., `np.linalg.LinAlgError`, `ValueError`). If a mathematical anomaly occurs (e.g., non-invertible innovation matrix), raise a custom domain exception (`CovarianceSingularityError`) that the API layer maps to a structured `500` or `422` HTTP response.

---

## 6. Testing Standards & Determinism

### A. Test Suite Architecture
*   **Unit Tests (`tests/unit/`):** Test isolated mathematical functions (e.g., passing a known 2D position vector and covariance into a Mahalanobis gating function and verifying the returned boolean flag against a hand-calculated reference value).
*   **Integration Tests (`tests/integration/`):** Test FastAPI endpoints using `httpx.AsyncClient` with `pytest-asyncio`, verifying complete multi-scan plot ingestion and JSON track responses.
*   **Property-Based Testing:** Where applicable, use the `hypothesis` library to generate randomized vectors and covariance matrices to assert invariants (e.g., asserting that Mahalanobis distance is always non-negative, or that Kalman covariance trace monotonically decreases after a measurement update).

### B. Mandatory Determinism
*   **Zero Randomness in Tests:** Automated test suites must be 100% deterministic and reproducible.
*   **Explicit Seeding:** Whenever generating synthetic radar plots, Gaussian noise, or mock tracks in tests, instantiate an explicit generator:
    ```python
    rng = np.random.default_rng(seed=42)
    noise = rng.multivariate_normal(mean=[0, 0], cov=R_matrix, size=10)
    ```
    Never use unseeded global state (`np.random.randn()`).

---

## 7. Git Workflow & Context Management

To maintain peak reasoning performance and prevent LLM context drift or token exhaustion across complex mathematical tasks, enforce disciplined session hygiene:

1.  **Atomic Commits:** After successfully passing the verification command for a micro-spec, stage and commit the changes using Conventional Commits:
    *   `feat(tracking): implement 2D constant velocity kalman predictor per Spec 01`
    *   `test(gating): add comprehensive chi-square boundary tests for Mahalanobis gate`
    *   `fix(pruning): resolve log-likelihood underflow in N-scan hypothesis evaluation`
2.  **Context Sanitization:** Once a commit is secured, explicitly instruct or remind the developer to clear or compact the session context (via `/compact` or opening a clean Claurst terminal session) before initiating the next specification ticket. Never carry implementation scratchpad memory from Task $N$ into Task $N+1$.

---

## 8. Summary Checklist Before Completing Any Task

Before declaring any micro-spec task complete, perform a self-audit against this checklist:
- [ ] Did I read the entire spec file and understand the exact boundary requirements?
- [ ] Are all matrix operations using stable linear algebra solvers (Cholesky/`scipy.linalg.solve`) rather than naive inverse?
- [ ] Are probabilities and hypothesis likelihoods computed and accumulated in log-space?
- [ ] Do all functions have complete type annotations and docstrings with matrix shape/unit specifications?
- [ ] Is the test suite 100% deterministic (explicit random seeds)?
- [ ] **Did I achieve a green build without modifying, weakening, or deleting a single test assertion?**
- [ ] Did the exact bash command specified in **Verification Command** exit with code `0`?
