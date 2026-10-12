# Interview narrative — use evidence, add your own experience

This public-data project is a portfolio case study. It does not claim you conducted Hillstrom's
experiment or shipped the resulting policy. Add genuine examples from your work below, respecting
confidentiality. Never invent company outcomes or personal responsibilities.

## A useful opening
“I built a reproducible workflow that separates whether a campaign works on average from whether
a model helps decide whom to contact. I chose the targeting rule on validation and estimated its
incremental value on randomized holdout data. The decision includes uncertainty and a simple
comparator; an uplift model is only useful if it improves that decision.”

Read the actual recommendation in `reports/report.md` before discussing the result.

## Product Analytics angle
- Define purchase as primary; explain why a visit increase alone does not establish value.
- Discuss SRM, feature balance, outcome logging and what the extract cannot verify.
- Distinguish absolute percentage-point ATE from relative uplift.
- Your experience: what experiment did you own, what quality issue did you find, and what changed?

## Product DS angle
- Explain why control conversion propensity is a strong but noncausal targeting comparator.
- Separate model training, policy selection and policy evaluation.
- Explain known assignment propensity, IPW, contact cost and contribution margin assumptions.
- Your experience: what business tradeoff, budget or stakeholder decision shaped your analysis?

## DS angle
- Compare S-learner pooling with T-learner flexibility and variance at rare conversions.
- Show pre-treatment feature whitelist and a leakage invariance test.
- Explain why Qini is a ranking diagnostic, not expected campaign profit.
- Your experience: what modeling failure taught you something, and how did you validate it?

## Questions to anticipate
Why not choose K on test? Why not sum predicted uplift? How would you power a prospective test?
What changes if assignment is observational? What if contact cost varies? What would make you
reject the model and use a simpler strategy? Why are current confidence intervals conditional?

Next extensions: repeated train splits for development (keeping a new final holdout), doubly
robust estimation with cross-fitting, revenue/margin modeling, multi-treatment policies and
prospective testing. Keep them out of the first version until they answer a concrete question.
