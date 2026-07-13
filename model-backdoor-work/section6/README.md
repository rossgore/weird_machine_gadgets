# Section 6 CLWE-RFF Construction

This package implements the sparse-secret CLWE random-Fourier-features
backdoor construction from Goldwasser et al., "Planting Undetectable
Backdoors in Machine Learning Models" (Section 6, Algorithms 5/6, Lemma 6.6,
Assumption 6.3).

## Directory structure

```
section6/
  core/
    __init__.py
    wd_gpd_sampler.py             Wd secret distribution + sparse embedding
                                   + LEGACY simplified GPd sampler
                                  
    clwe_rejection_sampler.py     TWO  GPd(bk) samplers:
                                   (a) CLWE-conditioned REJECTION
                                       sampler (Lemma 6.6) -- default;
                                   (b) Closed-form MIXTURE-OF-GAUSSIANS
                                       sampler (Bruna et al. STOC 2021,
                                       Def 2.19) -- sample_GPd_exact(),
                                       draws the exact hCLWE conditional
                                       density directly, no rejection band.
    section6_params.py            Derives (i, b, beta, tau, gamma) from
                                   Lemma 6.6's constraints given
                                   (d_sparse, m, target_delta)
    clwe_rff_model.py             CLWE_RFF_Model: earnest()/backdoored()
                                   constructors, branch-free forward(),
                                   activate_rff() (Algorithm 6, x' = x + bk),
                                   Assumption 6.3 margin diagnostic
    functional_indistinguishability.py
                                   Black-box, query-access-only
                                   indistinguishability checks -- logit
                                   distribution, prediction base rate,
                                   confidence-margin shape. Complements
                                   the weight-space (white-box) battery.
  tests/
    __init__.py
    test_full_battery.py           End-to-end test: derives
                                   params, builds both models, trains,
                                   fires the backdoor, runs weight-space
                                   (KS/Shapiro/Marchenko-Pastur on G) and
                                   functional-space indistinguishability
                                   checks.
  scripts/
    __init__.py
    run_demo.py                   Narrated wrapper around the battery. Best starting point.
    param_sweep.py                 Shows how (i, b, beta, tau) scale
                                   across target failure probabilities
                                   (parameter-derivation side only)
    verify_exact_sampler.py        Tries to validates the exact closed-form sampler:
                                   KS goodness-of-fit against its own
                                   analytic density, head-to-head vs the
                                   rejection sampler, and a backdoored
                                   model built with method="exact".
    battery_sweep.py               Sweeps the battery across a
                                   (D, d_sparse, m) grid: rebuilds fresh
                                   models at each point and re-runs the
                                   weight- and functional-space KS tests
                                   as a function of rho = d_sparse / D, to
                                   confirm the indistinguishability gap
                                   does not widen with sparsity. Coarse by
                                   default (~10 runs), fine-grainable via
                                   --grid.
  output/
    battery_results.txt            Plain-text results from the last run
    battery_sweep.csv              CSV formatted sweep grid results
    battery_sweep_results.txt      Full console log of the last sweep
    exact_sampler_verification.txt Full console log of verify_exact_sampler.py
    G_backdoored.npy / G_earnest.npy
  requirements.txt
```

## Running it

```bash
cd section6
pip install -r requirements.txt

python scripts/run_demo.py          # narrated end-to-end demo
python -m tests.test_full_battery   # CI-style, exit code 0 = all pass
python scripts/param_sweep.py       # optional: parameter scaling audit
python scripts/battery_sweep.py     # optional: sweep battery over d/D and m
python scripts/verify_exact_sampler.py  # optional: validate exact sampler
```

Finer sweeps:

```bash
python scripts/battery_sweep.py \
    --grid 64:5:1200 64:10:1200 64:20:1200 64:32:1200 \
           128:10:1200 128:20:1200 128:40:1200 \
    --n 6000 --epochs 800 --seed 7
```

### How to use samplers

Both samplers are exposed through the same constructor; the rejection
sampler stays the **default** so every existing result and test reproduces
unchanged:

```python
model, bk = CLWE_RFF_Model.backdoored(D, m, d_sparse, beta=..., tau=...)                  # rejection (default)
model, bk = CLWE_RFF_Model.backdoored(D, m, d_sparse, beta=..., tau=..., method="exact")   # closed-form
```

### What this does not do from Bruna et al. and Goldwasser et al.

This  does not implement the
full Bruna et al. (1) the worst-case-lattice-to-CLWE quantum
reduction, (2) the error-distribution smoothing arguments, and the (3) exact
discrete-Gaussian tail bounds over general lattices. The sampler draws
from Bruna's exact hCLWE density in closed form, but the hardness of
inverting is not shown.

## Extensions / next steps

- Functional-space checks here compare aggregate distributional behavior on
  CLEAN inputs. They do not attempt a full membership-inference-style
  attack, nor do they check behavior under adaptive/adversarial querying
  strategies designed specifically to detect the backdoor -- that would be
  a reasonable further extension.
