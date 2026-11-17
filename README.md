# Chest X-ray Pneumonia Classifier

> ⚠️ **EDUCATIONAL DEMO ONLY. NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE.**
> Do not use this to diagnose, screen, or make any decision about a real patient.

DenseNet121 transfer-learning classifier (NORMAL vs PNEUMONIA) for pediatric
chest X-rays, with Grad-CAM explanations, a FastAPI service, a Streamlit UI,
Docker, and GitHub Actions CI/CD.

Test set (evaluated once): ROC-AUC 0.982 · sensitivity 0.992 · **specificity 0.739**.
See [PLAN.md](PLAN.md) for the architecture, the pre-registered criteria, and
the full results, including the weaknesses.

```cmd
docker run -p 7860:7860 ghcr.io/hansteee/chest-xray-classifier:latest
```

Dataset: Kermany et al. (2018), Mendeley Data v2, doi:10.17632/rscbjbr9sj.2, CC BY 4.0.

*Full README coming in Phase 6.*
