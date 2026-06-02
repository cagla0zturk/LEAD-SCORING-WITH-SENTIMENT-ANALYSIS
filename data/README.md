# Data

## `raw/Leads.csv` — X Education Lead Scoring Dataset

* ~9,240 leads, 37 columns, binary target `Converted`.
* Source: Kaggle "Lead Scoring Dataset (X Education)". This repo ships a copy mirrored
  from a public GitHub copy of the same file. If the file is ever missing,
  `lead_priority.data.download.ensure_leads_csv()` re-downloads it (see
  `config.LEADS_CSV_URL`).
* The `"Select"` value in several categorical columns is an un-filled dropdown and is
  treated as missing (`NaN`) by `load_raw_leads`.

## `processed/` — generated artifacts (git-ignored)

Created by `python -m scripts.prepare_data`:

* `interactions.csv` — synthetic, **labelled** engagement notes (TR + EN) used to train
  the sentiment/intent classifier. See
  `lead_priority.data.synthetic_interactions` and the leakage discussion in the root
  `README.md`.
* `demo_leads.json` — a sample of real leads, each paired with a synthetic interaction
  note (chosen *independently* of `Converted`), used by `GET /leads/top`.
