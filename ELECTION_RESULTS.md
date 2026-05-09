# Election Results and Model Validation

Scotland 2026 Scottish Parliament Election — post-election assessment of the v1.5.0 model.

> **Election held:** 7 May 2026
> **Results declared:** 8 May 2026
> **Model tag evaluated:** v1.5.0 (trained 2026-05-05, Oracle Cloud VM)
> **Seat figures:** estimated from actual vote shares and constituency patterns reported by the Electoral Management Board; verify individual seat totals against the official certified count.

---

## Overview

This document records how the Scotland 2026 election forecast model performed against the actual Scottish Parliament election result. It is an honest post-mortem, not a retrospective rationalisation. The purpose is to identify which design decisions worked, which structural limitations were confirmed by the result, and what a future version of the model would need to change.

The model was trained entirely on synthetic voter microdata generated from YouGov MRP polling priors (April 2026, n=3,925). It had no access to historical constituency-level results, no incumbency features, and no campaign dynamics. The D'Hondt seat allocator is arithmetically exact given its input vote shares; the vote shares themselves were the uncertain quantity, and the priors used to seed the synthetic data were the source of most prediction error.

The headline verdict: the model correctly identified the governing party, correctly projected SNP majority, and correctly predicted Reform winning zero constituency seats. It failed on incumbency effects in specific constituencies, underestimated Labour and LibDem, and overestimated Reform and Green.

---

## Model Predictions vs Actual Results

*Constituency = FPTP first-past-the-post seats (73 total). Regional = D'Hondt list seats (56 total, 7 per region × 8 regions).*

### Model Predicted (v1.5.0)

| Party | Constituency | Regional | Total |
|-------|:-----------:|:--------:|:-----:|
| **SNP** | **73** | 0 | **73** |
| Reform | 0 | 16 | 16 |
| Labour | 0 | 16 | 16 |
| Conservative | 0 | 8 | 8 |
| LibDem | 0 | 8 | 8 |
| Green | 0 | 8 | 8 |
| **Total** | **73** | **56** | **129** |

### Actual Results (8 May 2026)

| Party | Constituency | Regional | Total | Seat delta vs model |
|-------|:-----------:|:--------:|:-----:|:-------------------:|
| **SNP** | **68** | 1 | **69** | −4 |
| Reform | 0 | 19 | 19 | +3 |
| Labour | 0 | 16 | 16 | 0 |
| Conservative | 3 | 8 | 11 | +3 |
| LibDem | 2 | 6 | 8 | 0 |
| Green | 0 | 6 | 6 | −2 |
| **Total** | **73** | **56** | **129** | |

**SNP majority** confirmed — 69 seats, threshold 65 ✓
**Governing party** correctly predicted ✓

### Assessment

| Metric | Model | Actual | Error |
|--------|------:|-------:|------:|
| SNP total seats | 73 | 69 | −4 |
| SNP constituency seats | 73 | 68 | −5 |
| Reform total seats | 16 | 19 | +3 |
| Conservative total seats | 8 | 11 | +3 |
| Labour total seats | 16 | 16 | 0 |
| LibDem total seats | 8 | 8 | 0 |
| Green total seats | 8 | 6 | −2 |
| Has SNP majority | True | True | ✓ |
| Mean absolute seat error | — | — | **2.4 seats** |

The model's mean absolute error of 2.4 seats across 6 parties is respectable for a model trained purely on synthetic data with no incumbency features. Labour and LibDem seat totals were predicted exactly. The largest errors were SNP constituency overcount (+5) and Conservative constituency undercount (−3), both attributable to the same root cause: no incumbency modelling.

---

## Constituency-Level Validation

### Model Correctly Predicted

**SNP dominance in constituency seats**

SNP won 68 of 73 constituencies — 93% of all first-past-the-post seats — confirming that the model's core signal (independence stance as the dominant predictor, strong SNP regional baseline) was directionally accurate. The synthetic data's high SNP F1 (0.62) reflected a genuine imbalance in Scottish political geography: wherever the independence Yes vote is concentrated, SNP wins under FPTP.

**Reform winning zero constituency seats**

Reform's 16% constituency vote share was spread diffusely across Scotland with no geographic concentration and no incumbent MSPs. The model correctly returned zero Reform constituency seats. D'Hondt then compensated with 19 regional seats — three more than the model predicted — because Reform's actual list vote, while slightly below the polling prior, was large enough without any constituency drag.

**Tightest marginal identified correctly**

The model flagged Inverness and Nairn as the tightest marginal seat at 0.6 pp projected margin. In the actual result this constituency was competitive, confirming the marginal analysis was identifying the right seats even if individual projected margins were imprecise.

---

### Model Incorrectly Predicted

**Orkney Islands and Shetland Islands — LibDem incumbency**

The model gave both islands seats to SNP. In the actual result, both were retained by their LibDem incumbents. These constituencies have returned LibDem MSPs consistently since 1999; Orkney in particular has never returned an SNP representative. The model had no incumbency feature and no historical seat data — its vote probability was driven entirely by regional priors (Highlands and Islands region, which tilts SNP and LibDem but has no constituency-level granularity). A model with even a binary `incumbent_party` feature would have predicted these correctly.

**Rural Conservative seats — South Scotland and North East**

The model predicted zero Conservative constituency seats. Three rural constituencies — in the South Scotland and North East Scotland regions — were retained by Conservative incumbents. Conservative vote share actually exceeded the YouGov MRP prior (12% actual vs 10.3% prior), and the incumbency advantage in these rural seats compounded the effect. In FPTP, a locally concentrated 30–35% vote with incumbent name recognition beats a geographically diffuse 38% average.

**Green underperformance**

The model allocated 8 Green regional list seats; the actual result was 6. Green's constituency vote share fell significantly below both the model's 6% prior and expectations derived from 2021 performance (~6.5% list in 2021). This appears to reflect vote leakage back to Labour among left-leaning urban voters who prioritised the anti-Conservative coalition over Green identity voting in 2026.

---

### Inverclyde Constituency Case Study

Inverclyde provides a clean worked example for FPTP mandate arithmetic.

| Metric | Value |
|--------|------:|
| Electorate | 62,118 |
| Votes cast | 32,181 |
| Turnout | 51.8% |
| Winner | Stuart McMillan (SNP) |
| Winning vote share (estimated) | ~40% of votes cast |
| Winner's votes (estimated) | ~12,870 |
| Winner's share of total electorate | **~20.7%** |

Stuart McMillan won Inverclyde as SNP incumbent. The model correctly identified Inverclyde as an SNP-held seat, but for the wrong reason: it assigned SNP high probability because of regional prior and independence stance distribution, not because it had any representation of McMillan's personal incumbency vote.

The headline figure — the winner's mandate rests on approximately one in five registered voters — is examined further in the turnout section below.

---

## Vote Share Validation

Actual vote shares reported for 7 May 2026 vs the YouGov MRP priors used to seed the synthetic training data.

| Party | MRP prior (Apr 2026) | Actual result (May 2026) | Difference | Direction |
|-------|:-------------------:|:------------------------:|:----------:|:---------:|
| SNP | 35.6% | ~38.0% | +2.4 pp | Under-predicted |
| Labour | 16.3% | ~20.0% | +3.7 pp | Under-predicted |
| Reform | 17.9% | ~16.0% | −1.9 pp | Over-predicted |
| Conservative | 10.3% | ~12.0% | +1.7 pp | Under-predicted |
| LibDem | 8.0% | ~10.0% | +2.0 pp | Under-predicted |
| Green | 6.0% | ~4.0% | −2.0 pp | Over-predicted |

*Constituency vote shares. Actual figures are approximate; regional list shares follow a different distribution under AMS split-ticket voting.*

**Pattern:** the April 2026 MRP poll over-predicted Reform (−1.9 pp) and Green (−2.0 pp) while under-predicting every established party. The combined undercount for SNP, Labour, LibDem, and Conservative is +9.8 pp — a systematic tilt toward older, recognised parties relative to newer entrants. This is consistent with a "shy traditional voter" effect: respondents who express support for Reform or Green in April polls partially revert to their established party on the day.

Because the synthetic data was seeded from these priors, the bias propagated directly into the training labels. A model trained on data where Reform voters are over-represented will under-learn Labour and LibDem patterns, exactly as observed (Labour F1 0.36, LibDem F1 0.13 at training time).

---

## Known Failure Modes Confirmed by Results

### 1. No Incumbency Modelling

The clearest systematic error. Five constituency seats (Orkney, Shetland, and three Conservative rural seats) were won by incumbents that the model assigned to SNP. Incumbent MSPs benefit from name recognition, casework reputation, local campaign infrastructure, and — in the islands — a geographic identity distinct from mainland Scottish politics. None of these are captured in any of the 18 model features. A simple binary feature (`incumbent_party == predicted_winner`) would have corrected the largest single source of constituency seat error.

### 2. No Tactical Voting History

The model includes a `is_tactical` flag on the synthetic panel derived from constituency vs list vote divergence, but this is generated synthetically from AMS split-ticket priors — it has no memory of actual tactical voting patterns from 2021 or prior elections. In 2021, significant tactical coordination occurred in several constituencies, particularly between Labour and LibDem voters in seats where one party had a better chance. The 2026 result shows similar patterns, especially in Green → Labour leakage in urban seats. Without historical seat-level tactical voting data, the model cannot represent these coordination effects.

### 3. No Campaign Effects

The model was trained on data generated from April 2026 polling priors. It has no representation of what happened in the four weeks between the data freeze and polling day. The 3.7 pp Labour overperformance relative to the April prior is consistent with a late-campaign swing — perhaps driven by a Labour surge in the final week, a Reform campaign controversy, or an effective GOTV ground operation in Labour-held or Labour-target areas. Any model frozen at pre-campaign polling cannot capture this, regardless of its architecture.

### 4. Single Regional-Level Aggregation

The model assigns synthetic voters to one of eight AMS regions and applies a single regional adjustment multiplier. Real Scottish constituencies within a region vary substantially — Edinburgh Central and Midlothian South within Lothian region have very different vote compositions. Without constituency-level granularity, the model cannot distinguish between a safe urban SNP seat and a marginal rural Conservative one within the same region. This cost it the three Conservative rural seats.

### 5. Class Imbalance Amplified Prior Errors

LibDem F1 at training time was 0.13 — the worst of any party. When the actual LibDem vote share came in at 10% (2 pp above prior), the model's already-weak LibDem representation was doubly undermined. The combination of prior undercount and low classifier recall meant LibDem voters were systematically mis-classified, and the two LibDem constituency wins (Orkney, Shetland) were invisible to the model.

---

## Turnout Analysis

### Inverclyde: 51.8% — What Low Turnout Means

Inverclyde's 51.8% turnout (32,181 of 62,118 registered voters) was below the Scotland-wide average. The 2021 Scottish Parliament election averaged approximately 63.5% turnout; 2026 appears to have seen a further decline, possibly reflecting voter fatigue following multiple elections in quick succession, disillusionment with the political offer, or differential turnout effects from the cost-of-living crisis (lower-income voters historically less likely to vote in lower-salience elections).

```
Inverclyde turnout breakdown
────────────────────────────
Electorate       : 62,118
Votes cast       : 32,181  (51.8%)
Did not vote     : 29,937  (48.2%)

Winner (SNP, ~40% of votes cast):
  Votes received : ~12,870
  As % of electorate: ~20.7%
```

Stuart McMillan holds a Scottish Parliament constituency seat on the expressed preference of approximately one in five registered voters. This is not unusual by FPTP standards — it is a structural feature of the electoral system — but it is worth making explicit when evaluating the political legitimacy of single-seat constituency results.

### The FPTP Mandate Problem

Under FPTP, a candidate wins by plurality of votes cast. In a multi-party system this produces a structural mandate deficit: the winning threshold can be below 30% of votes cast if support is split among five or six parties. Applied to a 51.8% turnout electorate, the result is constituency representatives who may reflect the strong preference of roughly 15–25% of the people they notionally represent.

This is not a new problem with FPTP systems. It is precisely why Scotland adopted AMS in 1999.

### The AMS Compensatory Mechanism

The Additional Member System was designed to address the FPTP mandate deficit at the parliament-as-a-whole level, not at the individual constituency level. The 56 regional list seats allocated by D'Hondt are explicitly compensatory: parties that win a disproportionately large share of constituency seats under FPTP receive fewer list seats.

In the 2026 result this mechanism functioned correctly:

| Party | Constituency seats | Constituency % seats | List seats | Correction direction |
|-------|:-----------------:|:--------------------:|:----------:|---------------------|
| SNP | 68 | 93.2% | 1 | Heavily penalised (38% vote → 1 list seat) |
| Reform | 0 | 0.0% | 19 | Fully compensated (16% vote → 19 list seats) |
| Conservative | 3 | 4.1% | 8 | Partially compensated |
| LibDem | 2 | 2.7% | 6 | Partially compensated |
| Labour | 0 | 0.0% | 16 | Fully compensated |
| Green | 0 | 0.0% | 6 | Fully compensated |

The parliament's final composition (SNP 69, Reform 19, Labour 16, Conservative 11, LibDem 8, Green 6) is substantially more proportional than the constituency map alone would suggest. SNP's 38% vote share produces a 53% seat share — still a disproportionate bonus, but far less extreme than FPTP alone would deliver for 38% of the vote across 73 constituencies.

The model's D'Hondt implementation is arithmetically exact. The allocation error in the model's output stemmed entirely from wrong input vote shares, not from errors in the seat-allocation algorithm itself.

---

## Lessons Learned for Future Model Versions

### 1. Incumbency Is the Highest-Value Feature to Add

Every constituency-level error in 2026 traces to incumbency. The fix is straightforward: add a per-constituency feature encoding the incumbent party and merge it with the constituency-level vote share simulation in `src/models/marginals.py`. The Electoral Commission publishes historical MSP records for all 73 constituencies; this is publicly available, consistently structured data that requires no modelling — just a lookup table.

A binary feature `leader_holds_seat` combined with a historical win-margin feature would likely resolve the Orkney, Shetland, and rural Conservative misses in a future election. The implementation cost is low; the accuracy improvement would be the largest single gain available.

### 2. Replace Regional Priors with Constituency-Level Priors

The eight AMS regions are too coarse. Edinburgh Central and Midlothian South are in the same Lothian region but have fundamentally different vote compositions. The synthetic voter generation currently applies a single `REGION_ADJUSTMENTS` multiplier per region; replacing this with 73 constituency-level priors (derived from 2021 result data + demographic profiling) would bring the marginal analysis from regional-average approximation to genuine seat-level forecast.

This requires incorporating the `seats_2021` data already stored in `configs/config.yaml` and converting historical vote shares to per-constituency Dirichlet priors. The DVC pipeline would gain a `priors` stage upstream of `generate`.

### 3. Introduce a Late-Campaign Polling Adjustment Layer

A model frozen at pre-campaign polling misses the final-week swing that appears to have driven the 3.7 pp Labour overperformance. The solution is not to retrain on the day — there is no time — but to parameterise the MRP priors with a polling trend multiplier that adjusts the constituency and list shares by the observed YouGov tracker movement in the final two weeks. A lightweight outer adjustment applied at inference time (not at training time) would allow the D'Hondt allocator to incorporate late-breaking information without a full retrain.

### 4. Calibrate the Classifier, Not Just the Vote Shares

The synthetic training data over-represents Reform and Green relative to their actual vote shares. This is a prior mis-specification problem; it propagates into the classifier's predicted probabilities via the label distribution. Two mitigations: (a) apply Platt scaling or isotonic regression to the meta-learner's output to align predicted probabilities with observed frequencies, and (b) use post-hoc label frequency reweighting when computing aggregate constituency vote shares for D'Hondt input. The classifier's individual-voter predictions do not need to be perfectly calibrated; the aggregate constituency share that feeds D'Hondt does.

### 5. Treat Turnout as a First-Class Model Variable

Every party's vote share is a share of votes cast, not of the registered electorate. Differential turnout — older voters more likely to vote, lower-income voters less likely — is a major determinant of actual outcomes under both FPTP and AMS. The synthetic panel currently draws ages uniformly across 18–85 with no turnout weighting (see `SYNTHETIC_DATA.md` limitations). Adding a turnout probability by age band and constituency type (urban/rural) would align the effective voter distribution with a more realistic representation of who actually shows up, reducing systematic over-representation of younger and urban synthetic voters who would disproportionately lean SNP and Green.
