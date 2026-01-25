# Crisis Funding Effectiveness Scoring Methodology

## Overview
We score humanitarian response plans based on how effectively funding reduced crisis severity over time. All components are normalized to 0-1 scale before combination.

## Three Components (Each Scored 0-1)

### Component 1: Severity Improvement Rate
**What it measures**: How much crisis severity decreased per month of intervention

**Raw calculation**:
```
Improvement_Rate_Raw = (INFORM_Start - INFORM_End) / Duration_Months
```

**Interpretation of raw value**:
- Positive = Crisis improved (severity decreased)
- Negative = Crisis worsened (severity increased)
- Example: 0.05 = severity dropped 0.05 points per month

**Normalization to 0-1**:
- Find best improvement in dataset (max positive change)
- Find worst worsening in dataset (max negative change)
- Map linearly: worst → 0, best → 1
- Crises that didn't change → 0.5

**Formula**:
```
Improvement_Rate_Normalized = (Improvement_Rate_Raw - MIN) / (MAX - MIN)
```

Where MIN = most negative improvement (worst crisis)
      MAX = most positive improvement (best crisis)

---

### Component 2: Consistency Score
**What it measures**: How steady the improvement was (vs. volatile ups and downs)

**Raw calculation**:
Standard deviation of INFORM scores across all months during intervention

**Interpretation of raw value**:
- 0 = Perfectly flat (no change at all)
- Low (0.1-0.3) = Steady improvement
- High (>1.0) = Erratic/volatile

**Normalization to 0-1**:
We want LOW variance to score HIGH, so we invert:

```
Consistency_Raw = 1 / (1 + Std_Deviation)
```

This gives a value between 0 and 1:
- If Std = 0 (perfectly flat) → Consistency = 1.0
- If Std = 1 (moderate volatility) → Consistency = 0.5
- If Std = 4 (very volatile) → Consistency = 0.2

**Already on 0-1 scale**, but we'll further normalize to ensure distribution:

```
Consistency_Normalized = (Consistency_Raw - MIN) / (MAX - MIN)
```

---

### Component 3: Cost-Effectiveness
**What it measures**: How much severity improvement per dollar spent

**Raw calculation**:
```
Cost_Effectiveness_Raw = Improvement_Rate_Raw / log(Funding_Per_Month + 1)
```

**Why log transform funding**: Diminishing returns - first $100M has more impact than next $100M

**Interpretation of raw value**:
- Higher = More improvement per dollar
- Can be positive (effective) or negative (ineffective)

**Normalization to 0-1**:
```
Cost_Effectiveness_Normalized = (Cost_Effectiveness_Raw - MIN) / (MAX - MIN)
```

---

## Composite Effectiveness Score

After normalizing all three components to 0-1:

**Effectiveness_Score = w1 × Improvement_Rate + w2 × Consistency + w3 × Cost_Effectiveness**

Where:
- w1 = 0.5 (50% weight on actual improvement)
- w2 = 0.3 (30% weight on consistency)
- w3 = 0.2 (20% weight on cost-efficiency)

**Result**: Score from 0-1 (we'll multiply by 100 for 0-100 display)

---

## "Good Crisis" Definition

**Threshold**: 66th percentile (top third)
- Crises scoring >= 66th percentile = "Successful interventions"
- These become training data for prediction model

**Rationale**: Top third represents genuinely effective responses while maintaining adequate sample size (target: 30+ examples for model training)
