# Product models

This directory contains every model distributed with LC-QSPR. Both model
families are maintained parts of the product:

- `CatBoost_phase.cbm` predicts phase existence from Mordred descriptors.
- `CatBoost_temperature_7000epochs.cbm` predicts transition temperatures from
  Mordred descriptors.
- `GATv2Conv_phase_temp_cv0.pt` jointly predicts phases and transition
  temperatures from molecular graphs.

The CatBoost and GNN inference workflows are demonstrated together in
`notebooks/05_predict.ipynb`.

These files are defaults rather than fixed requirements. The Python API and CLI
also accept explicit paths to CatBoost model pairs or individual GNN checkpoints
created by a user's own training run. Custom files can remain in a separate run
directory; they do not need to replace the distributed models.
