# Portfolio Migration Report

Migration date: 27 July 2026

Target repository: `lv19123/ml-portfolio`

Working branch: `feature/consolidate-ml-portfolio`

Draft pull request: `https://github.com/lv19123/ml-portfolio/pull/1`

## Repository Structure

```text
ml-portfolio/
├── README.md
├── PORTFOLIO_MIGRATION_REPORT.md
├── .gitignore
├── .gitattributes
├── assets/
│   └── portfolio/
├── classical-ml/
│   └── credit-scoring-system/
├── computer-vision/
│   └── person-background-replacement/
├── nlp-llm/
│   └── ai-lecture-meeting-summarizer/
└── ai-agents/
    └── student-ai-agent-telegram/
```

No nested Git repository or Git submodule is present.

## Preserved History

The three GitHub repositories were imported with `git subtree add` and without
`--squash`. Their original commits remain ancestors of the portfolio branch.

| Project | Original commits | Imported commits | History preserved |
|---|---:|---:|---|
| Person Background Replacement | 4 | 4 | Yes |
| AI Lecture / Meeting Summarizer | 2 | 2 | Yes |
| Student AI Agent for Telegram | 1 | 1 | Yes |
| Credit Scoring System | 0 | 1 | Not applicable: the local source had no `.git` directory |

Tree-object verification:

| Project | Original tree | Imported subtree tree | Match |
|---|---|---|---|
| Person Background Replacement | `f39a2640bf776756100ca0e10c7df656722c1b2a` | `f39a2640bf776756100ca0e10c7df656722c1b2a` | Yes |
| AI Lecture / Meeting Summarizer | `b2a6ae6dad612095d837aed53ee680675484e38e` | `b2a6ae6dad612095d837aed53ee680675484e38e` | Yes |
| Student AI Agent for Telegram | `d4ade9604dc337757be7b993b43d186f3492e3b2` | `d4ade9604dc337757be7b993b43d186f3492e3b2` | Yes |

The retained source histories cover:

- Student AI Agent: 24 June 2026;
- Person Background Replacement: 25 June 2026;
- AI Lecture / Meeting Summarizer: 28 June 2026.

Authors, author dates, commit dates, messages and original commit SHAs were not
rewritten.

## Files

| Project | Source | Target | Source files | Target files | Comparison | Git LFS |
|---|---|---|---:|---:|---|---|
| Person Background Replacement | `lv19123/person-background-replacement` | `computer-vision/person-background-replacement/` | 22 | 22 | Identical Git tree | One Keras model |
| AI Lecture / Meeting Summarizer | `lv19123/ai-lecture-meeting-summarizer` | `nlp-llm/ai-lecture-meeting-summarizer/` | 56 | 56 | Identical Git tree | Not used |
| Student AI Agent for Telegram | `lv19123/student-ai-agent-telegram` | `ai-agents/student-ai-agent-telegram/` | 22 | 22 | Identical Git tree | Not used |
| Credit Scoring System | Local completed project | `classical-ml/credit-scoring-system/` | 61 | 61 | Zero SHA-256 mismatches at import | Not required |

The Credit Scoring comparison was performed before the two documented
monorepo-path documentation changes below. Raw data, virtual environments,
caches, local credentials and ignored temporary files were not imported.

## Changes Inside Projects

No code, model architecture, dependency file, experiment output, image, video
or notebook was changed in the three imported GitHub projects.

Only these relocation-related changes were made inside a project:

- `classical-ml/credit-scoring-system/README.md`
  - replaced the placeholder clone command with the `ml-portfolio` URL;
  - changed the local `cd` path to the project directory in the monorepo;
  - changed the Google Colab clone and working-directory paths.
- `classical-ml/credit-scoring-system/PROJECT_STATUS.md`
  - documented that the pre-migration local source had no Git history;
  - changed the clone and working-directory paths to the monorepo.

Credit Scoring notebooks `01–08` were not executed, cleaned, reformatted,
resaved or modified. Their SHA-256 hashes after import are:

| Notebook | SHA-256 |
|---|---|
| `01_data_overview.ipynb` | `1d5ce48151fe28bc016fecab310dc3cb3f43d9039bc97259d0f09f1c2c977a0f` |
| `02_application_baseline.ipynb` | `8ebf53d3524ae2119384c7d79d2f5abbe1ce669038bfa647b6264facb5d09b89` |
| `03_bureau_features.ipynb` | `ec131db5e345029d8158aae780331b92f8657360dbd8735e1c088442438a18d6` |
| `04_previous_application_features.ipynb` | `f5974735cf1816761c740458dc28450679544ef8c4c0b93191ebe1717468b0cd` |
| `05_installments_features.ipynb` | `103caffe5a8554fb8ce6f7cdbad5ef09b8376b085bae0b263fd907e21119eea9` |
| `06_pos_cash_features.ipynb` | `0582ac73e9e06e4f1a8579d611ac2766fcfe926da2a456719854954d65d79244` |
| `07_credit_card_features.ipynb` | `644ae78297e2fbd3bc482c29156e50ecbe5f8006bea100e239bd24e848af7bb0` |
| `08_final_model.ipynb` | `f0d6c686f468f133c26d4bb49236bcea0f31e7676ff37b4cb98cd94f43ba26d8` |

## Validation

### Tests and syntax

| Check | Result |
|---|---|
| Credit Scoring `python3 -m pytest -q` | 62 passed, 1 skipped |
| Lecture / Meeting Summarizer `python3 -m pytest -q` | 44 passed, 1 dependency deprecation warning |
| Student AI Agent `python3 -m pytest -q` | 20 passed |
| Python `compileall` across all project source directories | Passed |

The skipped Credit Scoring test requires the raw Home Credit dataset, which is
intentionally absent from Git. The CV project has no automated test suite.
TensorFlow is not installed in the audit environment, so CV runtime inference
was not started. Its Python sources compile, and the saved Keras archive passes
an archive-integrity check.

### Credit Scoring CLI smoke test

- `python3 -m src.train --help`: passed;
- `python3 -m src.evaluate --help`: passed;
- `python3 -m src.predict --help`: passed;
- the saved model bundle loaded successfully;
- batch prediction produced two rows with client ID, default probability,
  predicted class and risk category;
- every generated probability was in the `[0, 1]` range.

No model training and no notebook execution was performed.

### Links, images and notebooks

- 31 relative Markdown link targets were checked;
- 13 embedded relative image targets were checked;
- no missing relative target was found;
- all nine Jupyter notebooks parse as valid JSON;
- existing PNG, JPEG and MP4 assets were retained.

The Person Background Replacement Streamlit URL responds with HTTP 303 and
redirects anonymous users to Streamlit authorization. It cannot currently be
verified as a public anonymous demo.

### Git LFS

- tracked path:
  `computer-vision/person-background-replacement/models/mobile_unet_model.keras`;
- object size: 236,488,039 bytes;
- object SHA-256:
  `02add9a2395e959609ff95d82c5400c7c94d60a8a3bb352d844a00af1a28bc9e`;
- `git lfs fsck`: passed;
- the object was uploaded to `lv19123/ml-portfolio`;
- a clean GitHub clone followed by `git lfs pull` downloaded the full object;
- the clean-clone size and SHA-256 match the source object.

No regular Git blob over 50 MiB is present in the final tree. The retained demo
videos remain below GitHub's per-file hard limit.

### Security and repository hygiene

- current files and every imported commit were scanned for common credential,
  token and private-key patterns;
- no credential was found;
- the one `sk-...` pattern match is a scikit-learn HTML CSS class stored in a
  notebook output, not a secret;
- `.env.example` files contain empty values, placeholders or non-secret
  defaults;
- `gitleaks` and `trufflehog` were not installed, so the scan used full-history
  filename checks and explicit high-confidence secret patterns;
- no tracked `.env`, key file, credential file, cache, virtual environment or
  IDE directory was found;
- no nested `.git` directory or submodule was found.

## Backup

Mirror backups with all refs were created before consolidation:

```text
/Users/maria/Desktop/Projects/ml-portfolio-backups-20260727/
├── person-background-replacement.git/
├── ai-lecture-meeting-summarizer.git/
└── student-ai-agent-telegram.git/
```

Git LFS objects were fetched into the mirror backup for the CV repository. The
old GitHub repositories were not deleted.

## GitHub

- repository: `lv19123/ml-portfolio`;
- visibility: public;
- branch: `feature/consolidate-ml-portfolio`;
- Draft PR: `https://github.com/lv19123/ml-portfolio/pull/1`;
- automatic merge: not performed;
- description: configured as requested;
- topics: `machine-learning`, `data-science`, `deep-learning`,
  `computer-vision`, `nlp`, `llm`, `rag`, `ai-agents`, `credit-scoring`,
  `python`, `portfolio`.

Important consolidation commits include:

- `9e663b7` — `chore: initialize machine learning portfolio`;
- `1b514f5` — CV subtree import;
- `de28be7` — NLP/LLM subtree import;
- `6dfe023` — AI Agent subtree import;
- `50a91ee` — `feat: add credit scoring system`;
- `2893262` — `docs: create portfolio overview`;
- `8e03f12` — `docs: update credit scoring monorepo paths`.

## Old Repositories

| Repository | Current visibility | Action | Reason |
|---|---|---|---|
| `lv19123/person-background-replacement` | Public | Not changed | Streamlit source/access must be migrated and verified first |
| `lv19123/ai-lecture-meeting-summarizer` | Public | Not changed | Draft PR must be reviewed and merged; GitHub CLI is unavailable |
| `lv19123/student-ai-agent-telegram` | Public | Not changed | Draft PR must be reviewed and merged; GitHub CLI is unavailable |

The repositories were not archived or deleted.

## Manual Actions Required

1. Review and merge Draft PR `#1` into `main` when satisfied.
2. In Streamlit Community Cloud, point the background-replacement deployment
   to repository `lv19123/ml-portfolio`, branch `main`, and entrypoint
   `computer-vision/person-background-replacement/app_streamlit.py`; then
   verify anonymous access and image/video inference.
3. Update any resume, application or saved links that still point to the three
   old repository URLs.
4. After the merged `main` branch and Streamlit deployment are verified, open
   each old repository on GitHub and use:
   `Settings` → `General` → `Danger Zone` →
   `Change repository visibility` → `Make private`.
5. Repeat the visibility change for all three old repositories. Do not delete
   them; keep the local mirror backups.

Until those manual cutover steps are complete, `ml-portfolio` is ready for
review but is not yet the only public repository.
