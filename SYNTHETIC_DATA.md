# Synthetic Voter Data — Design and Validation

Documentation for `src/data/generate_voters.py` and `src/features/pipeline.py`.

---

## Overview

No public voter microdata exists for Scottish Parliament elections. The Electoral Commission publishes aggregate results; YouGov and other pollsters publish headline vote shares and MRP constituency estimates; but individual-level voter records are not available as open data anywhere in the UK. This rules out training a model on observed voter behaviour.

The approach taken here is to generate a synthetic voter micro-panel whose statistical properties are anchored to the best available evidence: YouGov's April 2026 MRP poll (n=3,925, constituency-level estimates). Each synthetic voter is assigned demographic and attitudinal attributes drawn from plausible Scottish population distributions, then a vote intention is sampled from a per-voter probability vector that is derived from those attributes via multiplicative adjustments on the MRP regional priors.

This is not data augmentation and not a simulation of individual decision-making. It is a principled way to produce a labelled training dataset with realistic feature→label correlations — strong enough for a stacking ensemble to learn meaningful signal, honest enough to acknowledge that the signal is constructed rather than observed.

---

## YouGov MRP Polling Priors (April 2026, n=3,925)

These figures seed every aspect of the generation pipeline. They govern the baseline vote probability for each party before any individual feature adjustments are applied.

| Party | Constituency vote | Regional list vote |
|-------|:-----------------:|:-----------------:|
| SNP | 35.6% | 29.1% |
| Reform | 17.9% | 18.0% |
| Labour | 16.3% | 15.3% |
| Conservative | 10.3% | 11.1% |
| LibDem | 8.0% | 9.0% |
| Green | 6.0% | 8.5% |

**Independence polling (April 2026):** Yes 46.6% · No 44.6% · Undecided 8.9%

Independence stance is distributed across five ordered levels that sum to the headline Yes/No/Undecided split:

| Stance | Share |
|--------|------:|
| Strong Yes | 23.0% |
| Lean Yes | 23.6% |
| Undecided | 8.9% |
| Lean No | 22.3% |
| Strong No | 22.2% |

**Regional electorate weights** (used to assign voters proportionally to Scotland's 8 AMS regions):

| Region | Weight |
|--------|-------:|
| Glasgow | 16.0% |
| Lothian | 15.5% |
| Central Scotland | 13.5% |
| Mid Scotland and Fife | 13.5% |
| North East Scotland | 12.0% |
| South Scotland | 11.5% |
| Highlands and Islands | 8.5% |
| West Scotland | 9.5% |

---

## Generation Pipeline

The pipeline runs in four stages. Each stage feeds the next; none are re-run by DVC unless their upstream inputs change.

### Stage 1 — Demographic Generation

Draws the structural attributes of each synthetic voter from fixed probability distributions calibrated against Scotland-level census and survey benchmarks.

```python
# Region assignment — weighted by AMS electorate share
regions = rng.choice(REGIONS, size=n_voters, p=region_p)

# Age — uniform across 18–85 (no turnout weighting applied)
ages = rng.integers(18, 86, size=n_voters)

# Education — 6-level ordinal, calibrated to Scottish Household Survey
education = rng.choice(
    ["No qualifications", "Standard grades", "Highers",
     "HNC/HND", "Degree", "Postgraduate"],
    size=n_voters,
    p=[0.07, 0.18, 0.22, 0.12, 0.27, 0.14],
)

# Urban/Rural — 3-class split from NRS urban-rural classification
urban_rural = rng.choice(["Urban", "Suburban", "Rural"], p=[0.45, 0.35, 0.20])
```

Outputs per voter: `region`, `age`, `age_group`, `gender`, `education`, `urban_rural`.

---

### Stage 2 — Attitudinal Generation

Draws policy concern scores and political identity features. These are the attributes that drive vote intention in Stage 3.

```python
# Policy concern scores — independent Normal distributions, clipped to [0, 10]
economic_concern    = np.clip(rng.normal(6.5, 2.0, n_voters), 0, 10)
health_concern      = np.clip(rng.normal(6.8, 1.9, n_voters), 0, 10)
immigration_concern = np.clip(rng.normal(5.2, 2.5, n_voters), 0, 10)

# Top policy priority — determines which concern drives vote choice
top_priority = rng.choice(
    ["Economy", "Health", "Immigration", "Independence", "Housing"],
    p=[0.30, 0.28, 0.23, 0.07, 0.12],
)

# Independence stance — 5-level ordered scale anchored to MRP Yes/No split
independence_stance = rng.choice(
    ["Strong Yes", "Lean Yes", "Undecided", "Lean No", "Strong No"],
    p=[0.230, 0.236, 0.089, 0.223, 0.222],
)

# Previous vote — 2021 Scottish Parliament result (SNP 40%, Con 23%, Lab 22%, ...)
previous_vote = rng.choice(PREV_PARTIES, p=PREV_VOTE_PRIORS)
```

Outputs per voter: `economic_concern`, `health_concern`, `immigration_concern`, `top_priority`, `independence_stance`, `previous_vote`, `party_id_strength`, `nhs_satisfaction`, `cost_of_living_impact`.

---

### Stage 3 — Vote Intention

Computes a per-voter (6,) probability vector for both the constituency and regional list ballot, then samples a party from it. This is the core of the generation design.

```python
def _compute_vote_probs(regions, independence_stance, top_priority,
                        previous_vote, ages, base_priors, is_list_vote=False):
    # 1. Start from region-adjusted MRP prior
    probs = initialise_from_regional_prior(regions, base_priors)

    # 2. Apply multiplicative adjustments per feature
    adj *= INDEPENDENCE_ADJUSTMENTS[stance]   # strongest signal
    adj *= PRIORITY_ADJUSTMENTS[priority]     # top-issue party boost
    adj *= 2.5  # previous-vote loyalty (own party only)
    adj *= AGE_COHORT_ADJUSTMENTS             # old→Con/Reform, young→Green/SNP

    # 3. AMS split-ticket tendency (list vote only)
    if is_list_vote:
        adj[:, Green]  *= 1.2
        adj[:, LibDem] *= 1.2

    # 4. Row-normalise to valid probability simplex
    return (probs * adj) / row_sums
```

`constituency_vote` is the primary model target. `list_vote` is generated from the regional list priors with the same adjustments plus the AMS split-ticket boost. `is_tactical` flags voters whose two ballots differ, providing a diagnostic on tactical voting behaviour in the synthetic panel.

---

### Stage 4 — Validation

`get_vote_share_summary()` computes aggregate statistics immediately after generation so they can be compared against the MRP priors. The DVC `generate` stage prints this summary to stdout; CI captures it in the workflow log.

```python
summary = get_vote_share_summary(df)
# {
#   "constituency": {"SNP": 0.356, "Reform": 0.179, ...},
#   "list":         {"SNP": 0.291, "Reform": 0.180, ...},
#   "independence": {"Strong Yes": 0.230, ...},
#   "n_voters": 12500,
#   "tactical_rate": 0.182
# }
```

A tactical rate of ~18% is consistent with the AMS strategic voting pattern observed in 2021 (where Green and LibDem list votes notably exceeded their constituency shares).

---

## Feature Reference (18 Model Inputs)

The ML model is trained on 18 features drawn from the voter panel. Three are engineered post-generation; the remaining 15 are raw outputs from Stages 1–2.

| # | Feature | Type | Distribution / Values | Rationale |
|---|---------|------|-----------------------|-----------|
| 1 | `age` | Continuous | Uniform[18, 85] | Age is a top predictor of vote in UK elections; older voters trend Conservative/Reform, younger trend Green/SNP |
| 2 | `economic_concern` | Continuous | N(6.5, 2.0) clipped [0,10] | Core economic anxiety scale; high values boost Reform and Labour |
| 3 | `health_concern` | Continuous | N(6.8, 1.9) clipped [0,10] | NHS performance is the Scottish Government's most salient policy area |
| 4 | `immigration_concern` | Continuous | N(5.2, 2.5) clipped [0,10] | Higher variance reflects genuine polarisation on this issue |
| 5 | `tactical_swing_index` | Continuous (engineered) | [0, 1] | `abs(econ - health) / 10 × (4 - party_id) / 4` — captures swing likelihood when concerns diverge and partisan identity is weak |
| 6 | `indep_economy_interaction` | Continuous (engineered) | [-1, 1] | `indep_numeric × economic_concern / 10` — detects voters who support independence but punish the SNP government economically |
| 7 | `nhs_dissatisfaction` | Continuous (engineered) | [0, 8] | `(6 - nhs_satisfaction) × 8/5 × health_concern / 10` — inverted satisfaction scale weighted by health salience |
| 8 | `age_group` | Ordinal | 18-24, 25-34, 35-44, 45-54, 55-64, 65+ | Binned age for non-linear encoding; captures cohort effects beyond the continuous age signal |
| 9 | `party_id_strength` | Ordinal | 0–3 | Partisan attachment strength; low values amplify the tactical_swing_index |
| 10 | `nhs_satisfaction` | Ordinal | 1–5 | Raw NHS satisfaction (1=very dissatisfied); input to `nhs_dissatisfaction` engineering |
| 11 | `cost_of_living_impact` | Ordinal | 1–5 | Personal cost-of-living severity; higher values boost Reform and Labour |
| 12 | `education` | Ordinal | No quals → Postgraduate (6 levels) | Degree-educated voters trend Green/LibDem/SNP; lower education trends Reform/Conservative |
| 13 | `region` | Nominal | 8 AMS regions | Region is a structural confounder — South Scotland is Conservative heartland, Glasgow leans Labour and Green |
| 14 | `gender` | Nominal | Male, Female, Other | Included for completeness; weak signal in this synthetic panel |
| 15 | `urban_rural` | Nominal | Urban, Suburban, Rural | Rural → LibDem/Conservative in Highlands; Urban → Green/Labour in cities |
| 16 | `top_priority` | Nominal | Economy, Health, Immigration, Independence, Housing | Single most important issue; drives party-specific probability boosts |
| 17 | `independence_stance` | Nominal | Strong Yes → Strong No (5 levels) | Strongest individual predictor in the generation model; SNP multiplier 2.5× for Strong Yes |
| 18 | `previous_vote` | Nominal | SNP, Reform, Labour, Con, LibDem, Green, Did not vote | Vote loyalty captured by a 2.5× multiplier on the own-party probability |

**Note:** `list_vote` and `is_tactical` are present in the raw parquet but are excluded from `FEATURE_COLS` and are not model inputs. `voter_id` is a row identifier only.

---

## Probability Vector Construction

The core generation algorithm builds a (n_voters × 6) matrix of vote probabilities using multiplicative adjustments on the MRP regional priors. This is the mechanism that creates learnable signal in the dataset.

### Why multiplicative, not additive?

Additive adjustments risk producing negative probabilities for small parties. Multiplicative adjustments preserve the probability simplex after normalisation and behave correctly across all party sizes — a ×2.5 loyalty boost on a 6% prior stays proportional rather than overwhelming it.

### Regional prior initialisation

Before individual adjustments, each voter's probability vector is initialised from their region's adjusted MRP prior:

```python
REGION_ADJUSTMENTS = {
    "Glasgow":               {"Labour": 1.15, "Green": 1.20, "Reform": 0.90},
    "Highlands and Islands": {"SNP": 1.10,    "LibDem": 1.30, "Reform": 0.85},
    "South Scotland":        {"Conservative": 1.20, "Reform": 1.15, "SNP": 0.90},
    "Lothian":               {"Green": 1.25,  "LibDem": 1.15, "SNP": 0.95},
    # ... (8 regions)
}
```

These multipliers reflect documented regional variation in Scottish voting patterns and are applied before normalisation, so they shift the starting distribution without clamping any party to zero.

### Adjustment multiplier reference

| Feature value | Boosted parties | Suppressed parties |
|---------------|----------------|--------------------|
| Strong Yes (independence) | SNP ×2.5, Green ×1.2 | Conservative ×0.3, Labour ×0.5 |
| Lean Yes | SNP ×1.6 | Conservative ×0.6 |
| Strong No | Conservative ×1.8, Labour ×1.3 | SNP ×0.4 |
| Top priority: Immigration | Reform ×2.0, Conservative ×1.3 | SNP ×0.7 |
| Top priority: Independence | SNP ×2.0, Green ×1.4 | Conservative ×0.5 |
| Top priority: Health | Labour ×1.5 | — |
| Previous vote: own party | own ×2.5 | — |
| Age 65+ | Conservative ×1.4, Reform ×1.3 | Green ×0.5, SNP ×0.9 |
| Age 18–24 | Green ×1.5, SNP ×1.2 | Conservative ×0.4, Reform ×0.6 |

### The `dirichlet_concentration` parameter

The config carries `dirichlet_concentration: 50.0` and `generate_voters()` accepts it for API compatibility with an earlier version of the pipeline that added per-region Dirichlet noise on top of the regional priors. The current implementation replaces that noise layer with explicit regional multipliers (`REGION_ADJUSTMENTS`), which gives more interpretable and controllable regional variation than a concentration parameter. The parameter is retained to avoid breaking DVC stage fingerprints and CLI callers.

A concentration of 50 in the original design corresponded to a tight distribution around the prior mean — the synthetic regional shares would stay close to the MRP estimates with low variance. A value of 10–20 would have produced wider regional variation, risk overstating marginals in low-sample regions.

---

## Statistical Assumptions

| # | Assumption | Rationale | Limitation |
|---|-----------|-----------|------------|
| 1 | MRP constituency vote shares are unbiased priors | YouGov's April 2026 MRP (n=3,925) is the most methodologically rigorous pre-election estimate available | Polls can be systematically biased; MRP still mis-predicted 2019 UK election seat totals |
| 2 | Feature-vote relationships are multiplicative and independent | Multiplicative adjustments preserve the simplex and compose cleanly; avoids negative probabilities | Independence of adjustments ignores interaction effects (e.g. an economically anxious Strong Yes voter may behave differently than either factor predicts alone) |
| 3 | Independence stance is the dominant single predictor | Consistent with every YouGov Scottish cross-break since 2014; independence outranks party ID as a predictor | The relative salience of independence vs economic issues shifts across campaigns; a fixed multiplier overstates stability |
| 4 | Previous-vote loyalty is uniform across parties at 2.5× | A single loyalty multiplier simplifies a party-varying phenomenon (SNP loyalty ~70%, Conservative ~55% in 2021) | Uniform loyalty underestimates SNP retention and overestimates smaller-party retention |
| 5 | Age, education, and concern distributions match Scottish population | Education distribution calibrated to Scottish Household Survey; concern means set from British Election Study Scotland sub-sample | Synthetic distributions are marginals only — covariance structure between features (e.g. older voters having lower education) is not reproduced |
| 6 | The generation seed (42) produces a representative panel | Fixing the seed ensures reproducibility across DVC runs | A single seed is one draw from the distribution; results are not averaged over stochastic variation |

---

## Strengths

- **Fully reproducible** — fixed seed (42) and DVC content-addressed caching guarantee byte-identical output across runs on any machine with the same Python environment.
- **Realistic feature→label correlations** — multiplicative adjustment magnitudes are calibrated so the trained ensemble achieves SNP F1 ~0.62 and Labour F1 ~0.36, consistent with what a model trained on real survey data might produce given the feature set.
- **Covers both ballot papers** — constituency and list votes are generated from separate prior distributions with an AMS split-ticket adjustment, allowing the D'Hondt allocator to operate correctly on disaggregated vote shares rather than a single collapsed vector.
- **No licensing constraints** — entirely derived from published polling numbers; can be shared, reproduced, and extended freely without data agreements.
- **Diagnostic output built in** — `get_vote_share_summary()` produces an aggregate check against the MRP priors on every generation run, making it easy to detect if a code change has shifted the synthetic distribution unexpectedly.

---

## Limitations

### Demographic

- Age is drawn uniform[18, 85] with no turnout weighting. Real Scottish Parliament elections have substantially higher turnout among 55+ voters. The synthetic panel over-represents young voters relative to the actual electorate.
- Gender (Male/Female/Other at 49/49/2%) has no effect on vote probability in the generation model. In practice there is a small but consistent gender gap in Scottish voting, particularly on independence.
- Covariance between demographic attributes is not reproduced. Older voters in real data are correlated with lower formal education; in the synthetic panel these are sampled independently.

### Political

- The 2.5× previous-vote loyalty multiplier is uniform across all parties. Observed 2021 Scottish Parliament vote retention was approximately SNP 72%, Conservative 56%, Labour 55%, LibDem 51%. Uniform loyalty inflates retention for smaller parties.
- Region-level adjustments are fixed constants rather than distributions. They cannot capture within-region heterogeneity (e.g. Glasgow city vs. East Dunbartonshire within the Glasgow region).
- Late campaign effects, leader debates, and events after the April 2026 polling cut-off are not modelled. Any shift in vote intention between data generation and polling day is invisible to the trained model.

### Statistical

- Feature covariance structure is not reproduced — the synthetic panel draws each attitudinal variable independently, whereas real voters' views on the economy, immigration, and independence are correlated.
- A single random seed produces one realisation. Uncertainty about the synthetic distribution is not propagated into the model; confidence intervals on predictions are therefore narrower than they should be.
- The six concern/priority features drive vote intention via a small number of hand-coded multipliers. Any non-linear or threshold effects in real voter behaviour (e.g. immigration concern only becoming decisive above a threshold) are not captured.

### ML

- Because the training labels are generated from the same prior probabilities used to seed the feature distributions, the model cannot generalise to genuine out-of-distribution voters — it will always reflect the MRP priors rather than discovering new structure.
- Class imbalance (SNP ≈36% constituency share) suppresses per-class F1 for smaller parties. `class_weight=balanced` in the meta-learner partially compensates but cannot overcome a 6:1 imbalance ratio at the LibDem tail.
- There is no train/test split between polling regions or across time. All cross-validation is purely random and cannot detect regional or temporal generalisation failure.

---

## Post-Election Validation (8 May 2026)

The Scottish Parliament election took place on 8 May 2026 — the day after this document was authored. This section will record the model's predictive accuracy once official results are certified by the Electoral Management Board of Scotland.

### What the model predicted

| Party | Synthetic constituency share | Synthetic list share | Projected seats |
|-------|:---------------------------:|:--------------------:|:--------------:|
| SNP | 35.6% | 29.1% | 73 |
| Reform | 17.9% | 18.0% | 16 |
| Labour | 16.3% | 15.3% | 16 |
| Conservative | 10.3% | 11.1% | 8 |
| LibDem | 8.0% | 9.0% | 8 |
| Green | 6.0% | 8.5% | 8 |

### Actual results (pending)

| Party | Actual constituency share | Actual list share | Actual seats | Seat delta |
|-------|:------------------------:|:-----------------:|:------------:|:----------:|
| SNP | — | — | — | — |
| Reform | — | — | — | — |
| Labour | — | — | — | — |
| Conservative | — | — | — | — |
| LibDem | — | — | — | — |
| Green | — | — | — | — |

*Fill in once the Electoral Management Board certifies the full count.*

### Evaluation methodology

When results are available, validation will cover:

1. **Vote share error** — mean absolute error between synthetic constituency shares and actual first-preference shares across Scotland.
2. **Seat total error** — absolute difference in projected vs. actual seats per party.
3. **Marginal accuracy** — of the 56 constituencies flagged as marginal (< 5 pp margin), how many were genuinely competitive (margin < 5 pp in the actual result)?
4. **Directional accuracy** — did the model correctly identify the governing party?

The honest prior is that seat-level accuracy will be low: the model is trained on synthetic data derived from April 2026 priors and cannot account for campaign dynamics. The D'Hondt arithmetic is correct given the input vote shares; the vote shares themselves are the uncertain quantity.

---

## Future Improvements

### Short term (1–2 weeks)

- **Turnout weighting** — apply age-band turnout weights from the 2021 Scottish Parliament election to the uniform[18, 85] age distribution, correcting the over-representation of young voters.
- **Party-specific loyalty multipliers** — replace the uniform 2.5× retention factor with per-party values calibrated to 2021 Scottish Parliament vote-switching data (SNP ≈2.8×, Conservative ≈2.2×, Labour ≈2.2×).
- **Covariance structure** — use a Gaussian copula to reproduce observed correlations between age, education, and attitudinal variables from the British Election Study Scotland sub-sample.

### Medium term (1–2 months)

- **Real survey integration** — incorporate individual-level data from the publicly available British Election Study face-to-face wave (Scotland sub-sample, n≈800), using it to calibrate the adjustment multipliers empirically rather than setting them by hand.
- **Temporal dynamics** — add a campaign simulation layer that applies week-by-week polling trend lines to shift the MRP priors forward from April to May 2026, allowing the model to explore sensitivity to late swing.
- **Multilevel structure** — replace flat regional adjustments with a hierarchical model (constituency nested within region), giving each of the 73 constituencies its own prior rather than inheriting the regional flat prior.

### Long term (3+ months)

- **Actual voter data** — petition the Scottish Government or academic consortium (e.g. Scottish Social Attitudes Survey) for access to anonymised voter survey microdata under a data sharing agreement, replacing the synthetic panel with real observations.
- **Probabilistic seat projection** — replace the deterministic D'Hondt allocation with a Monte Carlo simulation over the full distribution of plausible vote share vectors, producing seat probability distributions rather than point estimates.
- **Continuous retraining pipeline** — wire the Evidently drift monitor to an automatic retrain trigger: when feature drift exceeds a threshold on incoming `/predict` request logs, enqueue a retrain job and swap the new model atomically via the `/model/retrain` endpoint.
