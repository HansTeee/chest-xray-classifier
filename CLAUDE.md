# CLAUDE.md: project rules

Chest X-ray pneumonia classifier (NORMAL vs PNEUMONIA), a capstone project.
Read `PLAN.md` for the architecture and the **pre-registered evaluation criteria**.

## Non-negotiable rules

1. **Never tune anything on the test set.** Thresholds, hyperparameters,
   model selection, early stopping, and augmentation choices are made on the
   validation set only. The test set is evaluated **once**, at the end.
2. **Every training run logs to MLflow** (`./mlruns`): params, per-epoch
   metrics, the confusion matrix and ROC curve (as image artifacts), and the
   model artifact.
3. **Every user-facing surface** (Streamlit UI, FastAPI OpenAPI docs, README,
   HF model/Space READMEs) shows a clear disclaimer: *educational demo, not a
   medical device, not for clinical use.*
4. **Commit after each completed phase** with a descriptive message.
5. **Model weights never go into git.** They live in the Hugging Face model
   repo and are downloaded at startup.
6. **Report metrics honestly, including the bad ones.** Report pass/fail
   against the pre-registered criteria in PLAN.md. Do not move the goalposts.

## Working conventions

- The user is on **Windows using CMD (not PowerShell)**. Any command the user
  must run themselves is given in CMD syntax (for example
  `.venv\Scripts\activate`, `set VAR=value`).
- Python 3.11 venv at `.venv/`.
  - Training deps: `requirements-train.txt` (CUDA 12.4 torch)
  - Serving deps: `requirements-serve.txt` (lean, CPU-only torch)
- Data lives in `data/chest_xray/{train,val,test}` (gitignored).
- Validation split is **patient-level** (no patient appears in two splits).
- No horizontal flips in augmentation.
- Tests must not need real weights or internet access (use the random-weight
  fixture).
- Lint with `ruff` before committing.
- Dataset attribution: Kermany et al., CC BY 4.0. Show it wherever sample
  images appear.
