# Evaluation

## Uploaded two-PDF benchmark

The policy benchmark contains 20 direct, multi-section, ambiguous, irrelevant, and out-of-document questions for `Leave-and-Holiday-Policy.pdf` and `DKT-Employee-Handbook-12.23.pdf`.

Run it against the completed PDFs in the configured local document library with real Ollama models:

```powershell
.\.venv\Scripts\python.exe -m eval.run_eval --questions eval\questions_uploaded_policies.json --uploaded-documents --out eval\results\uploaded_policies
```

The runner copies the source PDFs into a temporary evaluation workspace, so it does not add benchmark questions to the app's user Q&A history. It uses the real model to answer and judge coherence. User questions continue to use the normal lightweight evaluation path without a coherence judge.

Results are written to `eval/results/uploaded_policies/latest.json` and `latest.md`. The Insights dashboard shows this benchmark only while both source PDFs are present and their hashes match the evaluated files.
