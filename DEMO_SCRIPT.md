# 2-Minute Demo Script (Loom)

Total: **2:00**. Speaking pace: about 150 words per minute. Word counts per
section are sized to fit.

## Before you hit record (CMD)

```cmd
cd "C:\Users\Hans Tee\chest-xray-classifier"
docker run -d --name demo -p 7860:7860 -p 8000:8000 ghcr.io/hansteee/chest-xray-classifier:latest
.venv\Scripts\activate
mlflow ui --backend-store-uri ./mlruns
```

Open these tabs in order, then wait about 20 s for the container to download its weights:

1. `http://localhost:7860` (Streamlit UI, scrolled to top)
2. `http://localhost:8000/docs` (FastAPI Swagger)
3. README on GitHub, scrolled to the **Architecture** mermaid diagram
4. `http://127.0.0.1:5000` with experiment **chest-xray-pneumonia** open, all 3
   runs ticked, then **Compare** with the `val_auc` chart showing
5. GitHub **Actions**, showing the green run
   https://github.com/HansTeee/chest-xray-classifier/actions/runs/44072825585 with all 3 jobs expanded
6. README scrolled to the **Results vs. pre-registered criteria** table

Click "Use pneumonia 1" once before recording so the model is warm, then
reload the page.

After recording: `docker rm -f demo`

---

## 0:00 – 0:15 · Problem (15 s)

**Show:** Streamlit UI, top of page, red disclaimer banner visible.

> "Pneumonia is a leading cause of death in children under five, and reading
> chest X-rays takes expert time. I built an end-to-end pneumonia classifier,
> not just a model but a production-style ML system, while being explicit
> that it's an educational demo, not a medical device."

*(~45 words)*

---

## 0:15 – 0:50 · Live demo with Grad-CAM (35 s)

**Do:** Click **Use pneumonia 1**, then scroll to the result. Hover over the
probability bar, then the Grad-CAM image. Then click **Use pneumonia 2** and
scroll to the red "WRONG" banner.

> "I'll pick a sample from the held-out test set. The model predicts
> pneumonia, and the bar shows the probability against a threshold chosen on
> validation data. Every prediction comes with a Grad-CAM heatmap. Notice that
> here the hottest spot is the shoulder, not the lung. That's the kind of
> shortcut I flag rather than hide. And this second pneumonia case? The model
> misses it. It's one of three false negatives on the test set, and I left it
> in deliberately."

*(~85 words)*

---

## 0:50 – 1:15 · Architecture and FastAPI /docs (25 s)

**Show:** README mermaid diagram (about 8 s), then switch to `/docs`, expand
**POST /predict**, and scroll the response schema.

> "Training ran offline on my GPU. The weights live in a GitHub Release, not
> in git. At runtime a CPU-only Docker image runs FastAPI and Streamlit
> together, and downloads and hash-checks the weights at startup. The API has
> health, model-info and predict endpoints, input validation, and the
> disclaimer right in the docs."

*(~58 words)*

---

## 1:15 – 1:35 · MLflow comparison (20 s)

**Show:** MLflow compare view, `val_auc` per-epoch chart, then click into
`dn121_full_lr3e-4` → **Artifacts** → `test/test_confusion_matrix.png`.

> "Every run is tracked in MLflow: hyperparameters, per-epoch loss and AUC,
> confusion matrices and ROC curves. I compared three DenseNet variants with
> different learning rates and unfreezing depth, and selected on validation
> AUC only. The test set was touched exactly once."

*(~45 words)*

---

## 1:35 – 1:50 · CI/CD green run (15 s)

**Show:** GitHub Actions run with all three green jobs. Click **Docker build &
smoke test** and scroll to the `"label": "PNEUMONIA"` output.

> "Every push runs ruff, twenty-six pytest tests with no real weights, then a
> Docker build and a live smoke test against the real model. On main, the
> image is published to GitHub's container registry."

*(~37 words)*

---

## 1:50 – 2:00 · Results, limitations, disclaimer (10 s)

**Show:** README results table, then end on the red disclaimer banner.

> "Test AUC 0.98 and sensitivity 0.99, but specificity only 0.74. Single
> hospital, children only, not clinically validated. Not for clinical use."

*(~25 words)*

---

### Delivery notes
- Say the weak number (0.74) as clearly as the strong ones. That honesty is
  the point of the project.
- If you run long, trim the Grad-CAM sentence in the demo section first,
  never the disclaimer.
- Keep the cursor still while talking over static screens. Move it only to
  point at the threshold marker, the heatmap hotspot, and the green checks.
