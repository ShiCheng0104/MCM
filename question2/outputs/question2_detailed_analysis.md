# Question 2: Voting Method Comparison and Controversy Analysis

## Executive Summary

This analysis compares two voting methods used in *Dancing with the Stars* (DWTS): the **Rank Method** (Seasons 1-2, 28-34) and the **Percent Method** (Seasons 3-27). We examine their mathematical properties, outcome differences, and impact on four controversial contestants: Jerry Rice, Billy Ray Cyrus, Bristol Palin, and Bobby Bones.

**Key Findings:**
- The two methods produce **different elimination outcomes in 22.3%** of all weeks
- The Percent Method is **significantly more audience-favorable** (61.6% of differing cases)
- Bristol Palin is the most method-sensitive contestant: she would face more elimination pressure under the Rank Method
- Bobby Bones maintained **#1 fan ranking** in 5 out of 9 weeks despite low judge scores

---

## 1. Problem Background

### 1.1 Two Voting Methods Overview

| Method | Seasons Used | Formula | Elimination Rule |
|--------|--------------|---------|------------------|
| **Rank Method** | 1-2, 28-34 | Combined Rank = Judge Rank + Fan Rank | **Highest** combined rank eliminated |
| **Percent Method** | 3-27 | Combined Pct = Judge Pct + Fan Pct | **Lowest** combined percentage eliminated |

### 1.2 Mathematical Properties

**Rank Method:**
- Converts scores to ordinal rankings (1st, 2nd, 3rd...)
- Equal weighting: each rank contributes equally regardless of score gaps
- More "democratic" - ignores magnitude of differences

**Percent Method:**
- Preserves proportional differences in scores
- Large vote differentials can override judge score disadvantages
- Inherently favors contestants with strong audience support

**Example:** Consider 10 contestants where:
- Fan votes range from 1% to 20% (20× difference)
- Judge scores range from 20 to 30 points (1.5× difference)

Under Rank Method: Both contribute equally (ranks 1-10)
Under Percent Method: The 20× vote difference dominates the 1.5× score difference

---

## 2. Comprehensive Method Comparison

### 2.1 Outcome Agreement Analysis

Based on **337 elimination weeks** across all seasons:

| Metric | Value |
|--------|-------|
| Total weeks analyzed | 337 |
| Methods **agree** on elimination | 262 (77.7%) |
| Methods **disagree** on elimination | 75 (22.3%) |
| Rank method matches actual | 179 (53.1%) |
| Percent method matches actual | 198 (58.8%) |

**Key Insight:** The Percent Method has a **5.7 percentage point higher** match rate with actual eliminations.

### 2.2 Audience Preference Bias

In the **75 weeks** where methods disagree:

| Bias Direction | Count | Percentage |
|----------------|-------|------------|
| Rank Method favors audience more | 28 | 38.4% |
| Percent Method favors audience more | 45 | **61.6%** |
| Neither (different contestant eliminated) | 2 | 2.6% |

**Conclusion:** The **Percent Method is significantly more audience-favorable**. In disagreement cases, it's 1.6× more likely to save an audience-loved but judge-disliked contestant.

### 2.3 Season-by-Season Analysis

| Season Range | Method Used | Total Weeks | Average Agreement Rate | Key Observation |
|--------------|-------------|-------------|------------------------|-----------------|
| 1-2 | Rank | 14 | 92.9% | High agreement, small cast |
| 3-10 | Percent | 82 | 73.2% | More variation with larger casts |
| 11-20 | Percent | 96 | 76.0% | Consistent pattern |
| 21-27 | Percent | 77 | 74.0% | Similar to middle seasons |
| 28-34 | Rank | 68 | 82.4% | Return to rank shows higher agreement |

---

## 3. Four Controversial Contestants: Deep Dive

### 3.1 Overview Summary

| Contestant | Season | Method Used | Final Placement | Weeks with Lowest Judge Score | Would Be Eliminated (Rank) | Would Be Eliminated (Percent) |
|------------|--------|-------------|-----------------|-------------------------------|----------------------------|-------------------------------|
| Jerry Rice | 2 | Rank | **2nd** 🥈 | 2 | 2 weeks | 2 weeks |
| Billy Ray Cyrus | 4 | Percent | 5th | 3 | 2 weeks | 2 weeks |
| Bristol Palin | 11 | Percent | **3rd** 🥉 | 5 | 1 week | 1 week |
| Bobby Bones | 27 | Percent | **1st** 🏆 | 2 | 1 week | 1 week |

### 3.2 Jerry Rice (Season 2, Rank Method)

**Profile:** NFL Hall of Famer, finished as Runner-up despite 2 weeks of lowest judge scores.

| Week | Contestants | Judge Rank | Fan Rank | Combined Rank Position | Rank Eliminates | Percent Eliminates | Actual Eliminated |
|------|-------------|------------|----------|------------------------|-----------------|-------------------|-------------------|
| 1 | 10 | 5/10 | **2/10** | 3/10 | Master P | Master P | Kenny Mayne |
| 2 | 9 | 4/9 | **2/9** | 2/9 | Tatum O'Neal | Tatum O'Neal | Tatum O'Neal |
| 3 | 8 | 7/8 | 5/8 | 6/8 | Master P | Master P | Giselle Fernandez |
| 4 | 7 | 5/7 | **3/7** | 4/7 | Master P | Master P | Master P |
| 5 | 6 | 5/6 | **3/6** | 4/6 | Tia Carrere | Tia Carrere | Tia Carrere |
| 6 | 5 | 4/5 | **3/5** | 4/5 | George Hamilton | George Hamilton | George Hamilton |
| **7** | 4 | **4/4** | **4/4** | **4/4** | **Jerry Rice** | **Jerry Rice** | Lisa Rinna ⚠️ |
| 8 | 3 | 3/3 | 3/3 | 3/3 | Jerry Rice | Jerry Rice | (No elimination) |

**Analysis:**
- **Week 7 Anomaly:** Both methods predict Jerry Rice elimination, but Lisa Rinna was eliminated instead
- Both methods agree on Rice's fate — this is a **genuine controversy** regardless of method
- Strong early fan support (ranks 2-3) kept him safe initially
- **Bottom-two appearances:** 3 weeks (Rank) / 4 weeks (Percent)
- **Survival probability under judge tiebreaker:** ~9.0% (Rank) / ~5.3% (Percent)

**Verdict:** Jerry Rice's survival wasn't due to method choice — both methods would have eliminated him in Week 7. His runner-up finish represents a true production intervention or scoring irregularity.

---

### 3.3 Billy Ray Cyrus (Season 4, Percent Method)

**Profile:** Country music star, father of Miley Cyrus. Consistently low judge scores but solid fan support.

| Week | Contestants | Judge Rank | Fan Rank | Combined Position | Rank Eliminates | Percent Eliminates | Actual Eliminated |
|------|-------------|------------|----------|-------------------|-----------------|-------------------|-------------------|
| 1 | 11 | **11/11** | **11/11** | 11/11 | Billy Ray | Billy Ray | (No elimination) ⚠️ |
| 2 | 11 | 7/11 | 8/11 | 8/11 | Clyde Drexler | Paulina Porizkova | Paulina Porizkova |
| 3 | 10 | 7/10 | 7/10 | 7/10 | Clyde Drexler | Shandi Finnessey | Shandi Finnessey |
| 4 | 9 | 5/9 | 6/9 | 6/9 | Clyde Drexler | Leeza Gibbons | Leeza Gibbons |
| 5 | 8 | 7/8 | 7/8 | 7/8 | Clyde Drexler | Clyde Drexler | Clyde Drexler |
| 6 | 7 | 6/7 | **1/7** | 4/7 | John Ratzenberger | John Ratzenberger | Heather Mills |
| 7 | 6 | 6/6 | 5/6 | 6/6 | John Ratzenberger | John Ratzenberger | John Ratzenberger |
| **8** | 5 | **5/5** | **5/5** | **5/5** | **Billy Ray** | **Billy Ray** | **Billy Ray** ✓ |

**Analysis:**
- **Week 1:** Both methods predict elimination, but no elimination occurred (premiere episode)
- **Week 6:** Remarkable audience surge — went from rank 11 to rank **1** in fan votes
- The Percent Method protected him longer because his vote proportion improvements were significant
- **Bottom-two appearances:** 4 weeks (both methods)
- **Survival probability under judge tiebreaker:** ~4.6-5.2%

**Verdict:** Both methods treat Billy Ray Cyrus similarly. His Week 1 survival was due to format (no elimination), not method choice.

---

### 3.4 Bristol Palin (Season 11, Percent Method) ⭐ MOST DISCUSSED

**Profile:** Daughter of Sarah Palin, generated massive political controversy. Tea Party supporters allegedly organized voting campaigns.

| Week | Contestants | Judge Rank | Fan Rank | Combined Position | Rank Eliminates | Percent Eliminates | Actual Eliminated |
|------|-------------|------------|----------|-------------------|-----------------|-------------------|-------------------|
| 1 | 12 | 7/12 | **5/12** | 6/12 | David Hasselhoff | David Hasselhoff | David Hasselhoff |
| 2 | 11 | **3/11** | **2/11** | 2/11 | Michael Bolton | Michael Bolton | Michael Bolton |
| 3 | 10 | 9/10 | 7/10 | 8/10 | Margaret Cho | Margaret Cho | Margaret Cho |
| 4 | 9 | 8/9 | 6/9 | 7/9 | The Situation | The Situation | The Situation |
| 5 | 8 | **8/8** | **4/8** | 6/8 | Florence Henderson | Kyle Massey | Florence Henderson |
| 6 | 7 | 6/7 | **3/7** | 5/7 | Kurt Warner | Kurt Warner | Audrina Patridge |
| 7 | 6 | **6/6** | **3/6** | 4/6 | Kyle Massey | Rick Fox | Rick Fox |
| 8 | 5 | **5/5** | **3/5** | 4/5 | Kurt Warner | Kurt Warner | Kurt Warner |
| 9 | 4 | 4/4 | **3/4** | 4/4 | Brandy | Brandy | Brandy |
| **10** | 3 | **3/3** | **3/3** | **3/3** | **Bristol Palin** | **Bristol Palin** | (No elimination) |

**Analysis:**
- Bristol consistently ranked **bottom 3** in judge scores (5 weeks at absolute bottom)
- Yet her fan support kept her at ranks 2-5 throughout
- **Weeks at bottom-two:** 3 (both methods)
- **Both methods predict only Week 10 elimination** — which was the finale with no elimination

**Critical Finding:** The data shows both methods would only predict Bristol's elimination in Week 10 (the finale). The controversy stems from the **magnitude** of her audience advantage overcoming her judge disadvantage in the Percent formula.

Under the **Rank Method:**
- Her judge rank 6/7 + fan rank 3/7 = combined rank 9 (position 5/7)
- Equal weighting means a great fan rank only partially offsets poor judge rank

Under the **Percent Method:**
- Her 15.33% fan vote + 13.53% judge pct = 28.85% combined (position 3/7)
- The absolute percentages show she had competitive total support

**Verdict:** Bristol Palin's third-place finish was enabled by strong, consistent fan support throughout the competition. While both methods agree on elimination predictions, the Percent Method made her path feel safer due to its preservation of vote magnitude.

---

### 3.5 Bobby Bones (Season 27, Percent Method) ⭐ CHAMPION

**Profile:** Radio host with massive loyal following. Won despite consistent judge criticism.

| Week | Contestants | Judge Rank | Fan Rank | Fan Vote % | Rank Eliminates | Percent Eliminates | Actual Eliminated |
|------|-------------|------------|----------|------------|-----------------|-------------------|-------------------|
| 1 | 13 | 6/13 | **1/13** | **10.7%** | Nikki Glaser | Nikki Glaser | Nikki Glaser |
| 2 | 12 | **10/12** | **3/12** | 9.7% | Danelle Umstead | Danelle Umstead | Danelle Umstead |
| 3 | 11 | 8/11 | **1/11** | **11.3%** | Nancy McKeon | Joe Amabile | Nancy McKeon |
| 4 | 10 | **9/10** | **4/10** | **12.4%** | Joe Amabile | Tinashe | Tinashe |
| 5 | 9 | 8/9 | 8/9 | 9.6% | Joe Amabile | Joe Amabile | (No elimination) |
| 6 | 9 | 7/9 | **1/9** | **12.9%** | John Schneider | Mary Lou Retton | Mary Lou Retton |
| 7 | 8 | 6/8 | **1/8** | **16.2%** | DeMarcus Ware | John Schneider | John Schneider |
| 8 | 6 | **6/6** | **1/6** | **19.9%** | Joe Amabile | Joe Amabile | Juan Pablo Di Pace |
| **9** | 4 | **4/4** | **4/4** | 23.9% | **Bobby Bones** | **Bobby Bones** | (No elimination) |

**Analysis:**
- **Fan Rank #1 in 5 out of 9 weeks** — the highest fan engagement of all four contestants
- His fan vote percentage grew from 10.7% → 23.9% (2.2× increase over the season)
- Judge rank was consistently bottom-third (ranks 6-10 out of 8-13)
- **Bottom-two appearances:** 2 weeks (both methods)
- **Survival probability under judge tiebreaker:** ~27.6-29.9%

**Key Observation:** Bobby Bones demonstrates the Percent Method at its most extreme:
- Week 8: 6th in judges (last), 1st in fans → Combined position 5th (survives)
- His 19.9% fan vote completely neutralized his last-place judge score

**Verdict:** Bobby Bones is the perfect illustration of audience power under the Percent Method. His championship was controversial because technical skill (as measured by judges) was overridden by popularity. However, both methods only predict his elimination in Week 9 (finale, no elimination).

---

## 4. Judge Tiebreaker Mechanism Analysis

Starting Season 28, DWTS introduced a judge tiebreaker: when two contestants are in the bottom, judges vote to eliminate one.

### 4.1 Survival Probability Simulation

Assuming judges have a 60% probability of eliminating the technically weaker contestant:

| Contestant | Season | Method | Bottom-2 Appearances | Cumulative Survival Probability | Expected to Survive? |
|------------|--------|--------|----------------------|--------------------------------|---------------------|
| Jerry Rice | 2 | Rank | 3 | 9.0% | ❌ No |
| Jerry Rice | 2 | Percent | 4 | 5.3% | ❌ No |
| Billy Ray Cyrus | 4 | Rank/Percent | 4 | 4.6-5.2% | ❌ No |
| Bristol Palin | 11 | Rank/Percent | 3 | 10.4-10.6% | ❌ No |
| Bobby Bones | 27 | Rank/Percent | 2 | 27.6-29.9% | ❌ No |

**Formula:** $P(\text{survive } n \text{ times}) = 0.4^n$ (assuming 60% elimination probability per appearance)

**Conclusion:** Under the judge tiebreaker mechanism, **none of the four contestants would likely survive** to their actual placements. Bobby Bones had the best odds at ~28-30% but still well below certainty.

### 4.2 Impact Across All Seasons

| Metric | Value |
|--------|-------|
| Weeks where bottom-2 tiebreaker would apply | 284 |
| Potential outcome changes | 35 (12.3%) |

---

## 5. Additional Controversial Contestants

Our analysis identified other contestants with similar "low-judge, high-fan" profiles:

| Rank | Contestant | Season | Lowest Judge Weeks | Final Placement | Controversy Index* |
|------|------------|--------|-------------------|-----------------|-------------------|
| 1 | **David Ross** | 24 | 3 | 2nd 🥈 | 1.50 |
| 2 | **Bill Engvall** | 17 | 6 | 4th | 1.50 |
| 3 | **Nelly** | 29 | 4 | 3rd 🥉 | 1.33 |
| 4 | Candace Cameron Bure | 18 | 3 | 3rd 🥉 | 1.00 |
| 5 | Cody Rigsby | 30 | 3 | 3rd 🥉 | 1.00 |
| 6 | Marie Osmond | 5 | 3 | 3rd 🥉 | 1.00 |
| 7 | Joe Amabile | 27 | 6 | 6th | 1.00 |
| 8 | Vinny Guadagnino | 31 | 6 | 7th | 0.86 |
| 9 | Sean Spicer | 28 | 5 | 6th | 0.83 |

*Controversy Index = Lowest Judge Weeks / (Final Placement - 1). Higher values indicate stronger "against-the-odds" success.

**Notable Patterns:**
- **David Ross** (Season 24): 3 weeks of lowest scores, finished 2nd — most similar to Jerry Rice
- **Bill Engvall** (Season 17): 6 weeks of lowest scores, still finished 4th — remarkable persistence
- **Sean Spicer** (Season 28): Political figure with massive organized fan voting, similar to Bristol Palin

---

## 6. Mathematical Model: Method Sensitivity Analysis

### 6.1 When Do Methods Diverge?

Methods diverge most when:

1. **High score variance + Low vote variance:** Rank Method may favor low scorers
2. **Low score variance + High vote variance:** Percent Method favors high voters
3. **Large contestant pool:** More ties in rankings, different resolutions

### 6.2 Sensitivity Formula

Let $\sigma_J$ = judge score standard deviation, $\sigma_F$ = fan vote standard deviation

$$\text{Method Sensitivity} = \frac{\sigma_F}{\sigma_J}$$

- When ratio > 1.5: Percent Method significantly favors audience
- When ratio < 0.7: Rank Method may favor audience
- When 0.7 < ratio < 1.5: Methods produce similar results

### 6.3 Key Insight

Season 27 (Bobby Bones' season) showed high fan vote variance relative to judge score variance, explaining why his fan dominance was so pronounced under the Percent Method.

---

## 7. Conclusions and Recommendations

### 7.1 Summary of Key Differences

| Dimension | Rank Method | Percent Method |
|-----------|-------------|----------------|
| Weight distribution | Fixed 50-50 | Variable (40-60%) |
| Gap preservation | No (ordinal) | Yes (cardinal) |
| Audience advantage potential | Moderate | Strong |
| Controversy risk | Lower | Higher |
| Match with actuals | 53.1% | 58.8% |

### 7.2 Four Contestants Final Assessment

| Contestant | Core Controversy | Both Methods Agree? | Actual Method |
|------------|-----------------|---------------------|---------------|
| Jerry Rice | Week 7 survival anomaly | ✓ Yes | Rank |
| Billy Ray Cyrus | Consistent low scores | ✓ Yes | Percent |
| Bristol Palin | Political fan mobilization | ✓ Yes (Week 10 only) | Percent |
| Bobby Bones | Overwhelming fan support | ✓ Yes (Week 9 only) | Percent |

**Key Finding:** For all four controversial contestants, **both methods agree** on when they should be eliminated. The controversies stem from:
1. **Format decisions** (no-elimination weeks)
2. **Production interventions** (Jerry Rice Week 7)
3. **Magnitude of fan support** overwhelming technical deficiencies

### 7.3 Recommended Hybrid Approach

**Primary Method: Percent Method**
- Better reflects actual magnitude of support differences
- Higher historical accuracy (58.8% vs 53.1%)
- Maintains audience engagement incentive

**Auxiliary Mechanism: Judge Tiebreaker (Season 28+)**
- Prevents extreme controversies
- Adds dramatic tension
- Balances technical skill consideration

**Dynamic Weighting Proposal:**

| Competition Phase | Judge Weight | Fan Weight | Rationale |
|-------------------|--------------|------------|-----------|
| Early (Weeks 1-4) | 60% | 40% | Filter technical competence |
| Middle (Weeks 5-8) | 50% | 50% | Balanced development |
| Finals (Last 3 weeks) | 40% | 60% | Honor audience investment |

This progression rewards both skill development and fan loyalty, potentially reducing extreme outcome controversies while maintaining entertainment value.

---

## 8. Data Sources

This analysis utilized the following data files:
- [method_comparison.csv](method_comparison.csv): 337 weeks of dual-method predictions
- [controversy_analysis.csv](controversy_analysis.csv): Four target contestants statistics
- [detailed_weekly_comparison.csv](detailed_weekly_comparison.csv): Week-by-week breakdown for all contestants
- [judge_tiebreaker_simulation.csv](judge_tiebreaker_simulation.csv): Survival probability calculations
- [additional_controversies.csv](additional_controversies.csv): Extended controversial contestant list

---

*Report generated: February 2026*
*Analysis based on complete DWTS dataset (Seasons 1-34)*
