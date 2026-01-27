# Payments A/B Testing - Retry Logic Experiment

## Purpose
Design and evaluate a domain-agnostic A/B testing framework using a payments retry logic experiment as a case study.  
The goal is to demonstrate end-to-end experimentation thinking, including metric definition, guardrails, causal evaluation, and decision-making.

## Experiment Question
Does guided retry messaging increase completed payments after a soft decline compared to generic failure messaging, without increasing user friction?

## Hypothesis
Guided retry messaging (clear failure categorization and suggested next actions) increases final payment success and recovery rate after initial failure, without materially increasing retries or time-to-success.

## Experiment Design
- Unit of randomization: session-level
- Variants: Control (generic failure message) vs Treatment (guided retry messaging)
- Traffic split: 50 / 50

## Metrics
**Primary**
- Final payment success rate

**Secondary**
- Recovery rate after initial failure

**Guardrails**
- Retries per session
- Time-to-success
- Hard decline rate

## Status
Experiment design finalized.  
Synthetic data generation and analysis in progress.

## Key Takeaways
- Demonstrates end-to-end A/B experimentation design, from hypothesis to decision.
- Highlights the importance of guardrail metrics alongside primary uplift.
- Framework generalizes to marketing, sales, and operational experiments beyond payments.
