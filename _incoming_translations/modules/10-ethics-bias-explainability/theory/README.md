# Theory — Ethics, Bias & Explainability

## Why this module matters

AI models influence decisions about credit, hiring, criminal justice and health. Unfair, biased or opaque decisions have real consequences on human lives. This Module equips you to build responsible, fair and transparent systems.

______________________________________________________________________

## 1. Ethics in AI: Fundamentals

### Why do ethics matter?

- **Social impact:** AI affects opportunities, rights and well-being.
- **Legal liability:** Emerging regulations (EU AI Act, GDPR).
- **Reputation:** Unfair systems damage trust and brand.
- **Sustainability:** Unethical systems do not last.

### Fundamental principles

1. **Beneficence:** AI must benefit people.
1. **Nonmaleficence:** Do not cause harm.
1. **Autonomy:** Respect human decision.
1. **Justice:** Equitable distribution of benefits and risks.
1. **Explainability:** Decisions must be understandable.

📹 **Videos recommended:**

1. [AI Ethics Explained - MIT](https://www.youtube.com/watch?v=AaU6tI2pb3M) - 15 min
1. [Ethics of AI - Lex Fridman](https://www.youtube.com/watch?v=gmaONaP7TzI) - 30 min

📚 **Resources written:**

- [AI Ethics Guidelines - EU](https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai)

______________________________________________________________________

## 2. Types of bias in AI

### Sampling Bias

**Problem:** Dataset does not represent Objective population.

**Example:** Train disease detector with Images only from elite hospitals.

**Mitigation:**

- Stratified sampling.
- Verify demographic distribution.

### Historical Bias

**Problem:** Data reflects past discrimination.

**Example:** Historically trained hiring system where 90% of engineers were men.

**Mitigation:**

- Rebalance Data.
- Intervene in problematic features.

### Measurement Bias

**Problem:** Measurement method is systematically wrong for certain groups.

**Example:** Less accurate oximeters in dark skin.

**Mitigation:**

- Validate measurement instruments by subgroup.
- Use multiple data sources.

### Label Bias

**Problem:** Human labels contain prejudices.

**Example:** Moderators label minority content as "offensive" more frequently.

**Mitigation:**

- Multiple scorers.
- Audit of inter-scorer agreements.

### Aggregation Bias

**Problem:** Unique model for groups with different Features.

**Example:** Same Diabetes Prediction Model for all ethnicities (when risk factors differ).

**Mitigation:**

- Specific models by subgroup.
- Contextual features.

📹 **Videos recommended:**

1. [Bias in Machine Learning - Google](https://www.youtube.com/watch?v=59bMh59JQDo) - 8 min
1. [Understanding Fairness in ML - Microsoft](https://www.youtube.com/watch?v=jIU9JH9RsF0) - 15 min

______________________________________________________________________

## 3. Fairness

### Definitions of fairness

#### Demographic Parity

**Definition:** Rate of positive Predictions must be equal between groups.

**Example:** Credit approval % must be the same for men and women.

#### Equal opportunity (Equalized Odds)

**Definition:** True positive and false positive rates should be equal between groups.

**Example:** Hiring model must have the same success rate for qualified candidates from all groups.

#### Calibration

**Definition:** Predicted probabilities must reflect actual frequencies per group.

### Trade-offs

**Impossibility theorem:** All definitions of fairness cannot be satisfied simultaneously (except in trivial cases).

**Decision:** Choose definition of fairness according to context and stakeholders.

📹 **Videos recommended:**

1. [Fairness in ML - Moritz Hardt](https://www.youtube.com/watch?v=jIXIuYdnyyk) - 1 hour (fundamental)

📚 **Resources written:**

- [Fairness Definitions Explained](https://fairmlbook.org/)
- [AI Fairness 360 (IBM)](https://aif360.mybluemix.net/)

______________________________________________________________________

## 4. Explainability

### Why explainability?

- **Trust:** Users trust what they understand.
- **Debugging:** Identify Model Errors.
- **Compliance:** GDPR requires "right to explanation".
- **Justice:** Decisions that affect lives must be explainable.

### Global explanations

**Objective:** Understand general behavior of the Model.

**Methods:**

- **Feature Importance:** Which features are most important globally.
- **Partial Dependence Plots (PDP):** How Prediction changes when varying a feature.
- **Surrogate Models:** Simple (interpretable) Train Model that approximates a complex Model.

### Local explanations

**Objective:** Explain A specific Prediction.

**Methods:**

#### LIME (Local Interpretable Model-agnostic Explanations)

- Disturb input.
- Local linear Train Model around that Prediction.
- Interpretability of the linear model.

#### SHAP (SHapley Additive exPlanations)

- Based on game theory (Shapley values).
- Assigns contribution of each feature to the Prediction.
- Desirable theoretical properties (consistency, additivity).

**Usage:** SHAP is the de facto industry standard.

📹 **Videos recommended:**

1. [Explainable AI - StatQuest](https://www.youtube.com/watch?v=C80SQe16Rao) - 20 min
1. [SHAP Explained - Ritvik Math](https://www.youtube.com/watch?v=VB9uV-x0gtg) - 15 min
1. [LIME Explained - Krish Naik](https://www.youtube.com/watch?v=d6j6bofhj2M) - 20 min

📚 **Resources written:**

- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Interpretable ML Book (Free)](https://christophm.github.io/interpretable-ml-book/)

______________________________________________________________________

## 5. Interpretable models vs post-hoc explanations

### Inherently interpretable models

- **Linear/logistic regression:** Directly interpretable coefficients.
- **Decision Trees:** Clear visual logic.
- **Rules (if-then):** Transparent by design.

**Advantage:** Guaranteed interpretability.
**Disadvantage:** May sacrifice performance.

### Post-hoc explanations (black-box)

Explain complex Model (Random Forest, XGBoost, Neural Networks) after training.

**Tools:** SHAP, LIME.

**Advantage:** Do not sacrifice performance.
**Disadvantage:** Explanation is approximation, not absolute truth.

### Trade-off: accuracy vs Interpretability

It is not always necessary to sacrifice accuracy for interpretability. Try both approaches.

______________________________________________________________________

## 6. Risk and governance

### Impact evaluation

**Key questions:**

- Who is affected by the Model's decisions?
- What is the cost of an error (FP and FN)?
- Is there asymmetry of power (vulnerable users)?

### Documentation: Model Cards

**Content:**

- Purpose of the Model.
- Training data (distribution, limitations).
- Performance metrics by subgroup.
- Appropriate and inappropriate Usage cases.
- Ethical considerations.

📚 **Resources written:**

- [Model Cards Paper (Google)](https://arxiv.org/abs/1810.03993)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

### Continuous monitoring

- **Performance by subgroup:** Does the Model perform worse for any group?
- **Distributional shift:** Did the distribution of Predictions change?
- **Human feedback:** Do users report Problems?

### Appeal process

If the Model makes an adverse decision, there must be a human review process.

______________________________________________________________________

## 7. Real case studies

### COMPAS (Criminal Justice)

**Problem:** Recidivism Prediction Algorithm showed racial bias (more false positives for African Americans).

**Lesson:** Measure fairness by subgroup from the beginning.

### Amazon Recruiting Tool

**Problem:** Hiring system penalized CVs with the word "woman".

**Lesson:** Historical bias in Data is amplified.

### Facial Recognition

**Problem:** Commercial systems had much higher error rates in dark-skinned women.

**Lesson:** Evaluate performance in diverse subgroups.

📹 **Videos recommended:**

1. [AI Bias: Real Examples - Vox](https://www.youtube.com/watch?v=Ok5sKLXqynQ) - 10 min

______________________________________________________________________

## 8. Buenas Practices

- ✅ Include ethical review from the design phase (not at the end).
- ✅ Audit dataset: distribution, subgroups, possible biases.
- ✅ Measure fairness in Validation (not just global accuracy).
- ✅ Document decisions and trade-offs (Model Cards).
- ✅ Involve affected stakeholders in design.
- ✅ Implement explainability (SHAP) from development.
- ✅ Continuous monitoring of fairness in production.
- ✅ Establish human appeal process.

📚 **General resources:**

- [Fairness and Machine Learning (Book - Free)](https://fairmlbook.org/)
- [Google Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
- [Microsoft Responsible AI Resources](https://www.microsoft.com/en-us/ai/responsible-ai-resources)

______________________________________________________________________

## Final comprehension checklist

Before moving to the next Module, you should be able to:

- ✅ Identify Types of bias (sampling, historical, measurement, label) in dataset.
- ✅ Explain differences between demographic parity and equality of opportunity.
- ✅ Measure fairness for multiple demographic subgroups.
- ✅ Choose between global explanation (SHAP global) vs local (SHAP by Prediction).
- ✅ Implement SHAP to interpret complex Model.
- ✅ Document Model with Model Card.
- ✅ Propose concrete mitigation for detected bias.
- ✅ Design fairness monitoring process in production.

If you answered "yes" to all, you are ready to build responsible and ethical AI systems.
