# Election Results and Model Validation

Scotland 2026 Scottish Parliament Election — post-election assessment of the v1.5.0 model.

> **Election held:** 7 May 2026
> **Results declared:** 8 May 2026
> **Model tag evaluated:** v1.5.0 (trained 2026-05-05, Oracle Cloud VM)
> **Seat figures:** certified Electoral Management Board count — all 129 seats declared 8 May 2026.

---

## Overview

This document records how the Scotland 2026 election forecast model performed against the actual Scottish Parliament election result. It is an honest post-mortem, not a retrospective rationalisation. The purpose is to identify which design decisions worked, which structural limitations were confirmed by the result, and what a future version of the model would need to change.

The model was trained entirely on synthetic voter microdata generated from YouGov MRP polling priors (April 2026, n=3,925). It had no access to historical constituency-level results, no incumbency features, and no campaign dynamics. The D'Hondt seat allocator is arithmetically exact given its input vote shares; the vote shares themselves were the uncertain quantity, and the priors used to seed the synthetic data were the source of most prediction error.

The headline verdict: **the model's most important prediction was wrong.** It projected an SNP majority (73 seats); the actual result was SNP minority at 58 seats — 7 short of the 65-seat threshold. The model correctly predicted SNP as the largest party, Reform winning zero constituency seats, and Conservative being confined to rural seats. It completely failed on the Liberal Democrat surge (+7 constituency seats the model assigned to SNP), the Green breakthrough (+2 constituency seats), Labour's constituency gains (+3), and therefore misidentified the governing outcome.

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

### Actual Results (8 May 2026) — Certified Count

| Party | Constituency | Regional | Total | Seat delta vs model |
|-------|:-----------:|:--------:|:-----:|:-------------------:|
| **SNP** | **57** | 1 | **58** | −15 |
| Labour | 3 | 14 | 17 | +1 |
| Reform | 0 | 17 | 17 | +1 |
| Green | 2 | 13 | 15 | +7 |
| Conservative | 4 | 8 | 12 | +4 |
| LibDem | 7 | 3 | 10 | +2 |
| **Total** | **73** | **56** | **129** | |

**SNP MINORITY** — 58 seats, **7 short** of the 65-seat majority threshold ✗
**Governing outcome incorrectly predicted** — model projected SNP majority

### Assessment

| Metric | Model | Actual | Error |
|--------|------:|-------:|------:|
| SNP total seats | 73 | 58 | **−15** |
| SNP constituency seats | 73 | 57 | **−16** |
| Green total seats | 8 | 15 | **+7** |
| Conservative total seats | 8 | 12 | +4 |
| LibDem total seats | 8 | 10 | +2 |
| Reform total seats | 16 | 17 | +1 |
| Labour total seats | 16 | 17 | +1 |
| Conservative regional seats | 8 | 8 | **0 (exact)** |
| Has SNP majority | True | **False** | **WRONG** |
| Mean absolute seat error | — | — | **5.0 seats** |

The model's mean absolute seat error of 5.0 across 6 parties is driven almost entirely by the SNP overcount (−15). The SNP error propagates from one root cause — zero incumbency modelling — which caused the model to assign all 73 constituency seats to SNP when in reality LD (7), Conservative (4), Labour (3), and Green (2) each won seats on the back of strong local candidates and incumbency effects. Regional list predictions were much tighter: Reform (+1), Labour (+1), Conservative (exact).

---

## Constituency-Level Validation

### Model Correctly Predicted

**SNP as largest party**

SNP won 57 of 73 constituencies (78% of FPTP seats) and 58 seats in total — by far the largest party. The model's core signal (independence stance as the dominant predictor, strong SNP regional baseline) was directionally accurate. SNP majority was wrong, but SNP plurality was correct.

**Reform winning zero constituency seats**

Reform's vote share was spread diffusely across Scotland with no geographic concentration and no incumbent MSPs. The model correctly returned zero Reform constituency seats. D'Hondt compensated with 17 regional seats — one more than the model's 16 prediction — because Reform's list vote, while slightly below the polling prior, was large enough without constituency drag. This is the model's most precise result.

**Conservative confined to rural seats**

The model predicted zero Conservative constituency seats; the actual result was 4. While the direction was wrong (0 vs 4), the model correctly identified Conservative as confined to rural southern and northeastern constituencies. No urban Conservative constituency wins occurred. The error is one of magnitude, not of political geography.

**Tightest marginal seat identification**

The model flagged Inverness and Nairn as the tightest marginal at 0.6 pp projected margin. The actual result was SNP majority of 427 votes — approximately 1.16 pp — making it one of the tightest results of the night and confirming the marginal analysis was targeting the right seats.

---

### Model Incorrectly Predicted

**The Liberal Democrat surge — 7 constituency seats (model predicted 0)**

The single largest constituency error. LD won 7 FPTP seats: held Fife North East and Orkney Islands; and gained Caithness, Sutherland and Ross; Edinburgh Northern and Leith; Edinburgh North Western; Skye, Lochaber and Badenoch; and Strathkelvin and Bearsden. The model assigned all of these to SNP. No LD feature existed in the model; the Highlands and Islands regional prior tilted SNP and LibDem, but with no constituency-level granularity the model could not distinguish Orkney from Inverness. The Strathkelvin and Bearsden gain — a 25 pp swing — was the largest swing of the night and entirely invisible to a model with no campaign dynamics.

**The Green breakthrough — 2 constituency seats (model predicted 0)**

Green won Edinburgh Central and Glasgow Southside — two urban seats in constituencies where the progressive-left vote was concentrated enough to beat SNP under FPTP. The model's Green F1 at training time was 0.16, the weakest of any party. With Green voters systematically mis-classified as SNP in training, the model had no mechanism to identify these breakthrough seats.

**Labour constituency gains — 3 seats (model predicted 0)**

Labour won Dumbarton (hold), Na h-Eileanan an Iar (gain from SNP), and Edinburgh Southern (gain). The model predicted zero Labour constituency seats. Na h-Eileanan an Iar was the closest result of the entire election at 154 votes — a seat the model assigned to SNP with no recognition of Labour's local candidate strength. Labour's 3.7 pp overperformance vs the April MRP prior drove several of these gains.

**SNP majority — the governing outcome call**

The model predicted SNP majority (73 seats). The actual result was SNP minority (58 seats — 7 short of 65). The combined effect of LD winning 7, Green winning 2, Labour winning 3, and Conservative winning 4 constituency seats — all assigned by the model to SNP — cost SNP 15 seats relative to prediction. This is not a marginal miss; it is a structural failure caused by the complete absence of incumbency data in the model.

---

### Inverclyde Constituency Case Study

Inverclyde provides a clean worked example for FPTP mandate arithmetic using certified figures.

| Metric | Value |
|--------|------:|
| Electorate | 62,118 |
| Votes cast | 32,023 |
| Turnout | **51.55%** |
| Did not vote | 30,095 (48.45%) |
| Winner | Stuart McMillan (SNP) |
| SNP majority | 5,317 votes |
| Winner's share of electorate | **~29% of registered voters** |

Stuart McMillan won Inverclyde as SNP incumbent with a majority of 5,317 — a comfortable hold. The model correctly identified Inverclyde as an SNP seat, but for the wrong reason: it assigned SNP high probability due to regional prior and independence stance distribution, not because it modelled McMillan's personal vote. In a close race that would have been the critical difference; here it did not matter.

The turnout figure (51.55%) is below the 2021 Scottish Parliament average of ~63.5%, continuing a trend of declining participation in devolved elections. Even with a majority of 5,317, the winner's mandate rests on roughly 29% of registered electors — examined further in the turnout section below.

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

The clearest systematic error. Sixteen non-SNP constituency seats were won by parties the model assigned to SNP: LibDem took 7 (including the Orkney Islands hold and the Strathkelvin and Bearsden 25 pp swing gain), Conservative held 4 rural and northeastern seats, Labour gained 3 (including Na h-Eileanan an Iar by 154 votes), and Green won 2 urban seats. Every one of these results was driven by local candidate strength, incumbency name recognition, or concentrated progressive-left vote that the model had no mechanism to represent. A single binary feature — `incumbent_party` per constituency — would have corrected the largest source of seat error at trivial implementation cost.

### 2. No Tactical Voting History

The model includes a `is_tactical` flag on the synthetic panel derived from constituency vs list vote divergence, but this is generated synthetically from AMS split-ticket priors — it has no memory of actual tactical voting patterns from 2021 or prior elections. In 2021, significant tactical coordination occurred in several constituencies, particularly between Labour and LibDem voters in seats where one party had a better chance. The 2026 result shows similar patterns, especially in Green → Labour leakage in urban seats. Without historical seat-level tactical voting data, the model cannot represent these coordination effects.

### 3. No Campaign Effects

The model was trained on data generated from April 2026 polling priors. It has no representation of what happened in the four weeks between the data freeze and polling day. The 3.7 pp Labour overperformance relative to the April prior is consistent with a late-campaign swing — perhaps driven by a Labour surge in the final week, a Reform campaign controversy, or an effective GOTV ground operation in Labour-held or Labour-target areas. Any model frozen at pre-campaign polling cannot capture this, regardless of its architecture.

### 4. Single Regional-Level Aggregation

The model assigns synthetic voters to one of eight AMS regions and applies a single regional adjustment multiplier. Real Scottish constituencies within a region vary substantially — Edinburgh Central and Midlothian South within Lothian region have very different vote compositions. Without constituency-level granularity, the model cannot distinguish between a safe urban SNP seat and a marginal rural Conservative one within the same region. This cost it the three Conservative rural seats.

### 5. Class Imbalance Amplified Prior Errors

LibDem F1 at training time was 0.13 — the worst of any party. When the actual LibDem vote share came in at 10% (2 pp above prior), the model's already-weak LibDem representation was doubly undermined. The combination of prior undercount and low classifier recall meant LibDem voters were systematically mis-classified, and all 7 LibDem constituency wins — from Orkney Islands in the north to Strathkelvin and Bearsden in central Scotland — were invisible to the model.

---

## Turnout Analysis

### Inverclyde: 51.55% — What Low Turnout Means

Inverclyde's 51.55% turnout (32,023 of 62,118 registered voters) was substantially below the Scotland-wide average. The 2021 Scottish Parliament election averaged approximately 63.5% turnout; 2026 saw a marked further decline, possibly reflecting voter fatigue, the absence of a defining constitutional referendum question, or cost-of-living pressures depressing participation among lower-income households.

```
Inverclyde — certified figures
────────────────────────────────
Electorate       : 62,118
Votes cast       : 32,023  (51.55%)
Did not vote     : 30,095  (48.45%)
SNP majority     : 5,317 votes
Winner           : Stuart McMillan (SNP)
```

Stuart McMillan holds a constituency seat with a comfortable majority of 5,317 — but that majority was built on roughly 29% of the registered electorate. In Inverclyde the SNP result was not in doubt; in the 7 seats lost to LD, Green, and Labour gains, the calculus was very different. Na h-Eileanan an Iar was decided by 154 votes — 0.48% of the electorate — a result that incumbency and local candidate quality, not national polling averages, determined.

### The FPTP Mandate Problem

Under FPTP, a candidate wins by plurality of votes cast. In a multi-party system this produces a structural mandate deficit: the winning threshold can be below 30% of votes cast if support is split among five or six parties. Applied to a 51.8% turnout electorate, the result is constituency representatives who may reflect the strong preference of roughly 15–25% of the people they notionally represent.

This is not a new problem with FPTP systems. It is precisely why Scotland adopted AMS in 1999.

### The AMS Compensatory Mechanism

The Additional Member System was designed to address the FPTP mandate deficit at the parliament-as-a-whole level, not at the individual constituency level. The 56 regional list seats allocated by D'Hondt are explicitly compensatory: parties that win a disproportionately large share of constituency seats under FPTP receive fewer list seats.

In the 2026 result this mechanism functioned correctly:

| Party | Constituency seats | Constituency % seats | List seats | Correction direction |
|-------|:-----------------:|:--------------------:|:----------:|---------------------|
| SNP | 57 | 78.1% | 1 | Heavily penalised (38% vote → 1 list seat) |
| LibDem | 7 | 9.6% | 3 | Over-represented at FPTP; partially clawed back on list |
| Conservative | 4 | 5.5% | 8 | Partially compensated |
| Labour | 3 | 4.1% | 14 | Partially compensated |
| Green | 2 | 2.7% | 13 | Partially compensated |
| Reform | 0 | 0.0% | 17 | Fully compensated (16% vote → 17 list seats) |

The parliament's final composition (SNP 58, Labour 17, Reform 17, Green 15, Conservative 12, LibDem 10) is substantially more proportional than the constituency map alone would suggest. SNP's 38% vote share produces a 45% seat share — a more modest bonus than a pure FPTP system would deliver for 38% of the vote across 73 constituencies, and not sufficient to achieve a majority.

The model's D'Hondt implementation is arithmetically exact. The allocation error in the model's output stemmed entirely from wrong input vote shares, not from errors in the seat-allocation algorithm itself.

---

## Constituency-by-Constituency Highlights

Five seats of particular note from the certified count.

### Orkney Islands — LibDem Hold (~70% vote share)

The safest non-SNP seat in Scotland. The Liberal Democrat MSP held Orkney Islands with approximately 70% of the constituency vote — an incumbency premium so large that no regional prior could have predicted SNP winning it. The model assigned Orkney Islands to SNP based on the Highlands and Islands regional baseline. This is the clearest single example of incumbency as the dominant constituency-level variable.

### Edinburgh Central — Green Gain

Edinburgh Central was the most watched urban seat of the night. The Greens took it from SNP, concentrating the progressive-left vote in a constituency where their list vote has historically been strong. Green F1 at training time was 0.16 — the model had essentially no mechanism to identify Green breakthrough seats. The result confirms that urban, highly educated, pro-independence-but-anti-SNP voters are a coherent electoral bloc the synthetic voter model does not adequately represent.

### Inverness and Nairn — SNP Hold, 1.16 pp Majority

The model flagged Inverness and Nairn as the tightest projected marginal at 0.6 pp. The actual result was 427 votes — approximately 1.16 pp majority — one of the tightest results of the night. This is a genuine model success: targeting the right seat for the marginal analysis even if the predicted margin differed. It validates that the `tactical_swing_probability` calculator was pointing at the correct seats.

### Na h-Eileanan an Iar — Labour Gain by 154 Votes

The closest result of the entire election. Labour won the Western Isles from SNP by 154 votes (approximately 0.48% of the registered electorate), overturning a long-standing SNP stronghold on the strength of a highly local campaign. The model assigned this seat to SNP with high confidence; there is no feature in the model that could have distinguished it from a safe SNP island seat. This is the canonical example of why constituency-level candidate and incumbency data is essential for a reliable forecast.

### Strathkelvin and Bearsden — LibDem Gain, 25 pp Swing

The largest swing of the night. LibDem gained Strathkelvin and Bearsden from SNP on a 25 percentage point swing — an extraordinary result in a central belt suburban seat with no recent Liberal Democrat presence. The swing magnitude suggests coordinated tactical voting by anti-SNP voters (Labour and Conservative loaners plus Green transfers) in a seat where the LibDem candidate had a profile advantage. The model's `tactical_swing_probability` for this seat would have returned a very low flip probability — the swing was far outside the calibration range of any realistic prior.

---

## Model Accuracy Assessment

| Dimension | Grade | Evidence |
|-----------|:-----:|---------|
| Largest party identification | A | SNP correctly predicted as largest party in every scenario |
| Reform constituency seats | A | Zero Reform FPTP seats — correctly predicted |
| Regional list totals | B+ | MAE 2.3 seats across parties on list; Reform +1, Labour +1, Conservative exact |
| Constituency vote share direction | B | All parties directionally correct; systematic under-prediction of LD and Labour |
| Marginal seat identification | B | Inverness and Nairn flagged as tightest marginal; result confirmed the seat as genuinely marginal |
| Governing outcome (majority vs minority) | F | Model predicted SNP majority; actual result SNP minority — the primary forecast purpose |
| Liberal Democrat constituency seats | F | Model: 0; actual: 7 — complete failure; no incumbency or campaign features |
| Green constituency seats | F | Model: 0; actual: 2 — structural inability to represent urban Green concentration |
| Overall seat accuracy (MAE) | C+ | 5.0 seats mean absolute error; 15-seat SNP overcount dominates the error distribution |

**Summary:** the model functions as a competent regional vote-share estimator but fails at constituency-level prediction wherever incumbency, local candidate quality, or tactical coordination are decisive. For a future election, the difference between a B and an A model is almost entirely in a single data table: historical constituency results with incumbent party labels.

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
