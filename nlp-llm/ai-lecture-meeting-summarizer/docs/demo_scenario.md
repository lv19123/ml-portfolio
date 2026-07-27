# Demo Scenario

This scenario is designed for portfolio reviewers who want to see the main workflow quickly.

## Local Demo

1. Start the backend:

   ```bash
   uvicorn backend.app.main:app --reload
   ```

2. Start the frontend in another terminal:

   ```bash
   streamlit run frontend/app.py
   ```

3. Open the Streamlit UI:

   http://localhost:8501

4. Upload `sample_data/ml_lecture_sample.txt`.

5. Click `Process material`.

6. Generate `Short report`.

7. Generate `Full cleaned notes`.

8. Generate `Topics and pages/sources`.

9. Generate `Key terms`.

10. Build the RAG index.

11. Ask: `Для чего используется линейная регрессия?`

12. Download Markdown, PDF, and DOCX reports.

FastAPI docs are available at:

http://localhost:8000/docs

## Docker Demo

```bash
cp .env.example .env
docker compose build
docker compose up
```

Then open:

- FastAPI docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

The `data/` directory is mounted into the containers, so generated files remain available on the host machine.
