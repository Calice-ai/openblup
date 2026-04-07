# OpenBLUP – Known Issues

This file lists known bugs and design pitfalls in the codebase, formatted as GitHub issues.

---

## Critical Bugs

---

### [BUG] Double X'X computation in `MixedModelEquations::assemble` wastes work every REML iteration

**File:** `crates/core/src/lmm/mme.rs:60–75`

**Description:**

The `assemble` method first computes X'R⁻¹X with a hand-written O(nnz²)
triple-nested loop (lines 60–66), then immediately overwrites every entry it
just computed with the result of `compute_xtx_scaled` (lines 70–75). The first
loop produces **no observable effect** and runs on every REML iteration,
wasting CPU time.

**Relevant code:**

```rust
// lines 60–75 in mme.rs
// First loop – result is immediately discarded:
for (val, (row, col)) in x.iter() {
    for (val2, (row2, col2)) in x.iter() {
        if row == row2 {
            c[(col, col2)] += val * val2 * r_inv_scale;
        }
    }
}

// Overwrites everything the loop above computed:
let c_xtx = compute_xtx_scaled(x, r_inv_scale, n);
for i in 0..p {
    for j in 0..p {
        c[(i, j)] = c_xtx[(i, j)];
    }
}
```

**Reproducer:**
This is a correctness-neutral performance bug. To observe it, instrument both code paths and verify that removing the first loop produces identical `c` matrices:

```rust
// After fix: delete lines 60–66 and keep only compute_xtx_scaled.
// A test that checks the assembled matrix value is unchanged after the deletion
// confirms the first loop was dead.
```

**Expected fix:** Delete the triple-nested loop (lines 60–66). Keep only `compute_xtx_scaled`.

---

### [BUG] Per-animal `HashMap` in `compute_inbreeding` discards the cache between animals, causing O(n²) redundant work

**File:** `crates/core/src/genetics/ainverse.rs:276–280`

**Description:**
`relationship()` creates a fresh `HashMap` cache on **every call**. It is called once per animal in the inbreeding loop. The cache that accumulates values for animal _i_ is thrown away before computing animal _i+1_, so every shared ancestor is visited repeatedly. For a pedigree of _n_ animals this is O(n²) redundant recursive work instead of O(n²) total.

**Relevant code:**

```rust
// ainverse.rs:276–280
fn relationship(ped: &Pedigree, p: usize, q: usize, f: &[f64]) -> f64 {
    use std::collections::HashMap;
    let mut cache: HashMap<(usize, usize), f64> = HashMap::new();  // ← new cache every call
    relationship_cached(ped, p, q, f, &mut cache)
}
```

**Reproducer:**

```rust
// Build a deep line pedigree and measure time; should be noticeably super-linear.
// A correct implementation is O(n^2) total – the current one is O(n^3) for long chains.
let mut pedigree = Pedigree::new();
// add 200 animals in a chain: animal i has sire i-1
// then call compute_a_inverse(&pedigree) and measure wall time
```

**Expected fix:** Move the `HashMap` outside the per-animal loop and pass it across calls, or switch to the tabular A-matrix method (already available via `compute_a_matrix` in `hmatrix.rs`).

---

### [BUG] EM update for σ²_e in RR-BLUP is missing the PEV trace correction term

**File:** `crates/core/src/genetics/rrblup.rs:185`

**Description:**
The EM-REML update for `sigma2_e` (the residual variance) divides the residual sum of squares by `(n - p)` without the `tr(C⁻¹_{XX})` Prediction Error Variance correction. The correct formula is:

```
σ²_e(new) = (RSS + σ²_e · tr(C⁻¹_{ee})) / n
```

The `sigma2_u` update on line 177 **does** include its corresponding trace term. This asymmetry causes a biased estimator for `sigma2_e`, making variance ratio estimates unreliable.

**Relevant code:**

```rust
// rrblup.rs:177 – sigma2_u correctly includes trace:
sigma2_u = ((u_sum_sq + sigma2_e * trace_cinv_uu) / m as f64).max(1e-10);

// rrblup.rs:185 – sigma2_e is MISSING the analogous trace term:
sigma2_e = (resid / (n - p) as f64).max(1e-10);
// Should be: (resid + sigma2_e * trace_cinv_ee) / n
```

**Reproducer:**

```rust
// Fit a simple model where the true sigma2_e is known, and check that the
// estimate converges to the correct value.
// With the bug, the estimate will be systematically underestimated.
use openblup::genetics::RrBlup;
let genotypes = nalgebra::DMatrix::from_row_slice(10, 5, &[/* known data */]);
let x = nalgebra::DMatrix::from_element(10, 1, 1.0); // intercept only
let y = vec![/* known y with known variance */];
let mut rrblup = RrBlup::new(&genotypes, &x, &y);
let result = rrblup.fit(100, 1e-6).unwrap();
// result.sigma2_e will differ from the true value more than expected
```

**Expected fix:** Apply the full EM-REML update:

```rust
let trace_cinv_ee: f64 = (0..n).map(|i| c_inv[(i, i)]).sum();  // or the X-block diagonal
sigma2_e = ((resid + sigma2_e * trace_cinv_ee) / n as f64).max(1e-10);
```

---

### [BUG] `compute_g_matrix_auto` bypasses allele frequency length validation

**File:** `crates/core/src/gpu/mod.rs:44`

**Description:**
`compute_g_matrix_auto` calls the private `cpu_compute_g_matrix` directly, skipping the length check that the public `compute_g_matrix` in `gmatrix.rs` performs. If `allele_freqs.len() != marker_matrix.ncols()`, the inner loop silently reads from an out-of-bounds index (Rust will panic at runtime) or uses the wrong frequency for markers near the end of the array.

**Relevant code:**

```rust
// gpu/mod.rs:44 – no validation before calling cpu_compute_g_matrix
pub fn compute_g_matrix_auto(
    marker_matrix: &nalgebra::DMatrix<f64>,
    allele_freqs: &[f64],
) -> crate::error::Result<nalgebra::DMatrix<f64>> {
    // ...
    cpu_compute_g_matrix(marker_matrix, allele_freqs)  // ← no length check
}
```

**Reproducer:**

```rust
// This will panic (index out of bounds) at runtime instead of returning Err:
let markers = nalgebra::DMatrix::from_element(3, 5, 1.0); // 5 markers
let freqs = vec![0.5, 0.5];                               // only 2 freqs!
let _ = openblup::gpu::compute_g_matrix_auto(&markers, &freqs); // panic, not Err
```

**Expected fix:**

```rust
pub fn compute_g_matrix_auto(
    marker_matrix: &nalgebra::DMatrix<f64>,
    allele_freqs: &[f64],
) -> crate::error::Result<nalgebra::DMatrix<f64>> {
    if allele_freqs.len() != marker_matrix.ncols() {
        return Err(crate::error::LmmError::InvalidParameter(
            format!("allele_freqs length {} != marker columns {}", allele_freqs.len(), marker_matrix.ncols())
        ));
    }
    // ...
}
```

---

## High Severity

---

### [BUG] Convergence skipped on `iter == 0`, allowing premature exit after EM warm-up

**File:** `crates/core/src/lmm/ai_reml.rs:222`

**Description:**
The condition `if iter > 0 && rel_change < self.tol` skips the convergence check on the first AI iteration (index 0). If the EM warm-up steps already drove the parameters to near-convergence, the first AI step sees `rel_change < tol` but is **not** tested, and the _second_ iteration (index 1) fires the convergence check. This is fine in practice when EM and AI use the same iteration counter, but if the guard was intentionally meant to allow at least one full AI step, the correct check is `iter >= n_em_steps`.

**Relevant code:**

```rust
// ai_reml.rs:222
if iter > 0 && rel_change < self.tol {
    converged = true;
    break;
}
```

**Expected fix:** Use a minimum-AI-iterations guard:

```rust
if iter >= self.n_em_steps && rel_change < self.tol {
    converged = true;
    break;
}
```

---

### [BUG] Multi-level Wald F-test uses diagonal approximation instead of the correct quadratic form

**File:** `crates/core/src/diagnostics/wald.rs:102–125`

**Description:**
For terms with more than one degree of freedom, the Wald F-statistic should be the quadratic form `β' Var(β)⁻¹ β / k`. The current code approximates this as `(1/k) Σ(βᵢ / SEᵢ)²`, which ignores off-diagonal covariances. This approximation is incorrect for unbalanced or non-orthogonal designs, leading to wrong p-values.

The root cause is that `FitResult` only stores SE diagonals; the full `C⁻¹` submatrix is discarded before `wald_tests` is called.

**Relevant code:**

```rust
// wald.rs:114–125 – approximation acknowledged in the comment above:
let f_stat: f64 = indices
    .iter()
    .map(|&idx| {
        let ef = &result.fixed_effects[idx];
        if ef.se > 0.0 { (ef.estimate / ef.se).powi(2) } else { 0.0 }
    })
    .sum::<f64>()
    / num_df as f64;
```

**Expected fix:** Store the relevant `C⁻¹` submatrix in `FitResult` and use the full quadratic form.

---

### [BUG] Python binding panics the interpreter instead of raising `PyValueError` for non-contiguous allele-freq arrays

**File:** `crates/python-bindings/src/lib.rs:619–622`

**Description:**
Inside a `.map()` closure, `.expect()` is called on the result of `arr.as_slice()`. If the numpy array is non-contiguous (e.g. a slice or transposed array), this panics the Rust thread, which aborts the Python interpreter with no traceback instead of raising a `ValueError`.

**Relevant code:**

```rust
// lib.rs:619–622
let freqs_vec: Option<Vec<f64>> = allele_freqs.map(|arr| {
    arr.as_slice()
        .expect("Failed to read allele frequency array")  // ← panics Python
        .to_vec()
});
```

**Reproducer (Python):**

```python
import numpy as np
import openblup

markers = np.array([[0,1,2],[2,1,0]], dtype=np.float64)
# Non-contiguous array: every other element
freqs = np.array([0.5, 0.3, 0.5, 0.1])[::2]  # non-contiguous slice
# This will crash the Python interpreter, not raise ValueError
openblup.compute_g_matrix(markers, freqs)
```

**Expected fix:**

```rust
let freqs_vec: Option<Vec<f64>> = allele_freqs
    .map(|arr| -> PyResult<Vec<f64>> {
        Ok(arr.as_slice()
            .map_err(|e| PyValueError::new_err(format!("allele_freqs must be contiguous: {}", e)))?
            .to_vec())
    })
    .transpose()?;
```

---

### [BUG] `set_data` in Python bindings inserts columns in non-deterministic order

**File:** `crates/python-bindings/src/lib.rs:217–220`

**Description:**
The `columns` parameter is typed as `HashMap<String, PyObject>`, which has random iteration order. Columns are added to the `DataFrame` in an unpredictable sequence, which changes between runs and Python versions. Code downstream that accesses columns by position (instead of by name) will silently use the wrong data.

**Relevant code:**

```rust
// lib.rs:217–220
fn set_data(&mut self, py: Python<'_>, columns: HashMap<String, PyObject>) -> PyResult<()> {
    let mut df = DataFrame::new();
    for (name, obj) in &columns {  // ← random iteration order
```

**Reproducer (Python):**

```python
import openblup
model = openblup.Model()
# Column insertion order is non-deterministic across runs:
model.set_data({"y": [1.0, 2.0], "x": [3.0, 4.0], "z": [5.0, 6.0]})
# In different runs, columns may be added as y,x,z or z,y,x or any permutation
```

**Expected fix:** Change `HashMap` to `IndexMap` (already a dependency via `nalgebra`/`sprs` transitive deps, or add `indexmap` explicitly):

```rust
fn set_data(&mut self, py: Python<'_>, columns: IndexMap<String, PyObject>) -> PyResult<()> {
```

---

### [BUG] `c_inv` field is always `Some(...)` but typed as `Option`, causing unnecessary `unwrap()` calls

**File:** `crates/core/src/lmm/mme.rs:184`, `ai_reml.rs:938`

**Description:**
`MmeSolution::c_inv` is documented as "None if not computed" but `MmeSolution::solve` always sets it to `Some(chol.inverse())`. The `Option` wrapper is dead. Downstream code in `ai_reml.rs:938` calls `.as_ref().unwrap()` — if a future refactor makes `c_inv` legitimately `None`, this silently panics with no error message.

**Relevant code:**

```rust
// mme.rs:184
c_inv: Some(c_inv),  // ← always Some

// ai_reml.rs:938
let c_inv = sol.c_inv.as_ref().unwrap();  // ← bare unwrap in production code
```

**Expected fix:** Change the field type to `nalgebra::DMatrix<f64>` (non-optional) and update all call sites.

---

### [BUG] `variance_se` silently returns all-zeros when the AI matrix is singular

**File:** `crates/core/src/lmm/ai_reml.rs:877–883`

**Description:**
When Cholesky decomposition of the Average Information matrix fails (e.g. at a boundary estimate), `variance_se` returns `vec![0.0; n_params]`. Callers cannot distinguish "SE is exactly zero because the estimate is on a boundary" from "SE could not be computed due to a numerical failure". This will propagate zeros as valid SEs into the final `FitResult`.

**Relevant code:**

```rust
// ai_reml.rs:877–883
match ai.clone().cholesky() {
    Some(chol) => {
        let ai_inv = chol.inverse();
        (0..n_params).map(|i| ai_inv[(i, i)].abs().sqrt()).collect()
    }
    None => vec![0.0; n_params],  // ← silent failure
}
```

**Expected fix:** Return `f64::NAN` for failed SEs, or propagate a `Result`:

```rust
None => vec![f64::NAN; n_params],
```

---

## Medium Severity

---

### [PERF] Dense C⁻¹ matrix is allocated and stored on every REML iteration without size guard

**File:** `crates/core/src/lmm/mme.rs:175`

**Description:**
`MmeSolution::solve` calls `chol.inverse()` to store the full `(p + Σqᵢ) × (p + Σqᵢ)` dense matrix every iteration. For 10 000 genotypes the MME dimension is ~10 003 and `c_inv` consumes ~800 MB per iteration. There is no size check, warning, or lazy-computation option.

**Expected fix:** Add a size guard or make `c_inv` computed on demand:

```rust
const MAX_DENSE_DIM: usize = 2000;
if self.dim > MAX_DENSE_DIM {
    return Err(LmmError::InvalidParameter(
        format!("MME dimension {} too large for dense C⁻¹ (limit {}). Use sparse solver.", self.dim, MAX_DENSE_DIM)
    ));
}
```

---

### [PERF/BUG] `compute_a_matrix` returns a dense O(n²) matrix with no memory limit

**File:** `crates/core/src/genetics/hmatrix.rs:41–66`

**Description:**
`compute_a_matrix` allocates a full `n × n` dense `DMatrix<f64>`. For 100 000 animals this is 80 GB. The function is public and called by H-matrix code without any size guard.

**Expected fix:** Add an explicit size limit and error:

```rust
const MAX_ANIMALS_DENSE: usize = 5000;
if n > MAX_ANIMALS_DENSE {
    return Err(LmmError::InvalidParameter(
        format!("Pedigree has {} animals; dense A-matrix would require {}GB. Use sparse A⁻¹ instead.", n, (n*n*8) / 1_000_000_000)
    ));
}
```

---

### [BUG] EM-score approximation in AI-REML has no test against analytical scores

**File:** `crates/core/src/lmm/ai_reml.rs:618–639`

**Description:**
The REML score vector is derived by performing an EM step and back-computing via `score_k = (q_k / 2σ²_k²) · (σ²_k_em − σ²_k)`. The code comment claims this is exact, but it is only exact to first order in `Δσ²`. For poorly conditioned systems the approximation may slow or prevent convergence. No unit test compares these scores against the analytical first derivative of the log-likelihood.

---

### [BUG] `ConvergenceMonitor` is defined but not used — AI-REML duplicates its logic inline

**File:** `crates/core/src/diagnostics/convergence.rs`, `crates/core/src/lmm/ai_reml.rs:194–225`

**Description:**
`ConvergenceMonitor` provides `record()`, `is_converged()`, and `max_reached()`, but `ai_reml.rs` re-implements the same convergence check inline. The struct is dead library code. Either use it in `ai_reml.rs` or remove it.

---

### [BUG] Potential off-by-one in `col_off2` for models with 3+ random terms

**File:** `crates/core/src/lmm/mme.rs:110–122`

**Description:**
When assembling cross-terms `Z_k'R⁻¹Z_l` (l > k), `col_off2` is initialized to `p` inside the **outer** `k` loop but the inner `l` loop advances it. For k=0 this works correctly; for k=1+, `col_off2` starts at `p` but the `l` loop iterates from 0, meaning the `col_off2 + j` index written for l=0 overlaps with the X-block columns. This can silently corrupt C for models with three or more random terms.

**Relevant code:**

```rust
// mme.rs:110–122
let mut col_off2 = p;  // ← reset to p for every k, but inner loop doesn't skip k's own block
for (l, z2) in z_blocks.iter().enumerate() {
    if l != k && l > k {
        let cross = compute_xtz_scaled(z, z2, r_inv_scale, n);
        for i in 0..z.cols() {
            for j in 0..z2.cols() {
                c[(row_offset + i, col_off2 + j)] = cross[(i, j)];
                c[(col_off2 + j, row_offset + i)] = cross[(i, j)];
            }
        }
    }
    col_off2 += z2.cols();
}
```

**Reproducer:**

```rust
// A model with 3 random terms should be tested for C matrix symmetry and
// correct block structure:
let mme = MixedModelEquations::assemble(&x, &[z1, z2, z3], &y, 1.0, &[ginv1, ginv2, ginv3]);
for i in 0..mme.dim {
    for j in 0..mme.dim {
        assert_relative_eq!(mme.coeff_matrix[(i,j)], mme.coeff_matrix[(j,i)], epsilon=1e-10);
    }
}
// This test may fail due to the off-by-one in cross-term placement.
```

---

### [BUG] Python API always uses `Identity::new(1.0)` — user-specified variance structures are silently ignored

**File:** `crates/python-bindings/src/lib.rs:400`

**Description:**
When constructing random terms from Python, the variance structure is hardcoded to `Identity::new(1.0)` regardless of any user input. The `VarStruct` trait implementations (AR1, Diagonal, Unstructured, FactorAnalytic) are completely inaccessible from the Python API.

**Relevant code:**

```rust
// lib.rs:400
builder = builder.random(&rt.column, Identity::new(1.0), rt.ginv.clone());
//                                    ^^^^^^^^^^^^^^^^^ always Identity
```

**Expected fix:** Expose variance structure selection in `RandomTermPy` and map it to the correct Rust type before calling `.random()`.

---

### [BUG] No `NaN`/missing-value handling — `NaN` in response vector silently corrupts all estimates

**File:** `crates/core/src/data/dataframe.rs`, `crates/core/src/lmm/ai_reml.rs`

**Description:**
`DataFrame` stores `f64` columns with no `NA` concept. If the response vector `y` contains `NaN` values (common in real breeding datasets), the mean computation, residual sum of squares, and log-likelihood calculations all propagate `NaN` silently. The REML engine will run to completion and return `NaN` variance estimates without any error or warning.

**Reproducer:**

```rust
// Build a model where y contains NaN and observe that fit_reml returns NaN estimates
// rather than an error.
let y = vec![1.0, 2.0, f64::NAN, 4.0];
// ... assemble and fit: variance estimates will be NaN
```

**Expected fix:** Validate the response vector at model build time:

```rust
if y.iter().any(|v| v.is_nan() || v.is_infinite()) {
    return Err(LmmError::InvalidParameter("Response vector contains NaN or Inf values".into()));
}
```

---

## Low Severity / Code Quality

---

### [REFACTOR] Final solve in `rrblup.rs` is redundant — last iteration already holds the converged solution

**File:** `crates/core/src/genetics/rrblup.rs:198–209`

**Description:**
After the EM convergence loop, the code assembles and solves the MME a second time using the converged parameters. The last loop iteration already computed this identical solution. The duplicated solve wastes time proportional to the MME dimension.

**Expected fix:** Save the solution from the last loop iteration and break without re-solving.

---

### [REFACTOR] `sparse_to_dense` is defined identically in two files

**File:** `crates/core/src/lmm/mme.rs:349`, `crates/core/src/lmm/ai_reml.rs:~1009`

**Description:**
Both files contain a function `sparse_to_dense` with identical behavior. This violates DRY and means any future bug fix must be applied in two places.

**Expected fix:** Move the function to `crates/core/src/matrix/sparse.rs` and import it in both files.

---

### [REFACTOR] `build_result` recomputes log-likelihood and fitted values that were already computed in the iteration loop

**File:** `crates/core/src/lmm/ai_reml.rs:900–989`

**Description:**
The log-likelihood and fitted values (`Xb + ΣZu`) are recomputed from scratch in `build_result` even though they were calculated during the final convergence iteration. Pass them as arguments to avoid redundant matrix multiplications.

---

### [REFACTOR] Pedigree cycle detection runs twice — `validate()` and `sort_pedigree()` both run Kahn's algorithm

**File:** `crates/core/src/genetics/pedigree.rs`

**Description:**
Both `validate()` and `sort_pedigree()` independently run Kahn's topological sort algorithm to detect cycles. Users who call `validate()` followed by `sort_pedigree()` pay twice. The documentation does not mention that `sort_pedigree()` subsumes cycle detection.

**Expected fix:** Have `validate()` call `sort_pedigree()` internally, or document that they are redundant and one should not call both.

---

### [DOCS] `pyproject.toml` is missing Python version classifiers and project URLs

**File:** `pyproject.toml`

**Description:**

- No `Programming Language :: Python :: 3.x` classifiers despite `requires-python = ">=3.8, <3.14"`.
- Author entry `{name = "Jakob"}` is missing an email field.
- No `homepage` or `repository` URL in project metadata.

This causes the package to appear incomplete on PyPI and makes it harder for users to find the source repository.

---

### [REFACTOR] `RrBlup::new` uses `assert!` (panic) for dimension checks instead of returning `Result`

**File:** `crates/core/src/genetics/rrblup.rs:60–61`

**Description:**
Dimension mismatches in `RrBlup::new` cause a panic rather than returning an `Err`. All other constructors in the library use `Result` for validation. This inconsistency makes the API unsafe to call from library code.

**Relevant code:**

```rust
// rrblup.rs:60–61
assert_eq!(n, y.len(), "genotype rows must match y length");
assert_eq!(n, x.nrows(), "X rows must match y length");
```

**Reproducer:**

```rust
// This panics instead of returning Err:
let genotypes = nalgebra::DMatrix::from_element(5, 3, 1.0);
let x = nalgebra::DMatrix::from_element(3, 1, 1.0); // wrong row count
let y = vec![1.0; 5];
let _ = openblup::genetics::RrBlup::new(&genotypes, &x, &y); // panic!
```

**Expected fix:**

```rust
pub fn new(genotypes: &DMatrix<f64>, x: &DMatrix<f64>, y: &[f64]) -> Result<Self> {
    let n = genotypes.nrows();
    if n != y.len() {
        return Err(LmmError::InvalidParameter(format!(
            "genotype rows ({}) must match y length ({})", n, y.len()
        )));
    }
    if n != x.nrows() {
        return Err(LmmError::InvalidParameter(format!(
            "X rows ({}) must match y length ({})", x.nrows(), n
        )));
    }
    // ...
}
```

---

### [TEST] No numerical regression tests validate estimates against reference implementations

**File:** `crates/core/tests/`

**Description:**
The test suite checks structural properties (symmetry, sign, dimension) and trivial cases but contains no end-to-end tests comparing AI-REML variance estimates against known-correct values from ASReml, R's `sommer`, or `lme4`. For a statistics library, correctness of numerical output is the most critical property and is currently untested.

**Expected fix:** Add at least one integration test using a small, published dataset with known REML estimates (e.g. from a textbook example) and assert that estimates match to within a reasonable tolerance (e.g. `1e-4`).
