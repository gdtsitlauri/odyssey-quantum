# Limitations

- Quantum evaluation is limited to small simulator-friendly circuits.
- Public datasets lack native post-quantum transition metadata, so fragility augmentation is assumption-driven.
- On the current public `UNSW-NB15` adapter, the strongest standalone Odyssey result selects the classical zero-uncertainty path, so there is no public-data quantum advantage claim.
- The strongest Odyssey-family public benchmark path is the validation-stacked ensemble, but it still remains slightly below `RandomForest` on PR-AUC.
- Laptop-feasible defaults favor reproducibility and modest scale over benchmark saturation.
- This repository is a research prototype, not a production intrusion detection platform.
