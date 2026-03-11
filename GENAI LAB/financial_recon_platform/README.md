# GenAI Financial Reconciliation Platform

Production-oriented reference implementation of a modular reconciliation platform using:

- FastAPI for APIs
- PostgreSQL + pgvector for storage and vector search
- Rule-based + embedding-based + LLM-assisted matching

## Quick start

1. Copy `.env.example` to `.env` and update values.
2. Start services:

```bash
docker compose up -d --build
```

3. Run tests:

```bash
pytest -q
```

4. Run demo script:

```bash
python scripts/demo_reconcile_bank_gl.py
```

## Implemented scenarios

- Bank statement <-> GL cash accounts (detailed strategy)
- Customer payments <-> AR invoices (detailed strategy)
- Other scenarios wired via extensible strategy interface and constraints hooks

## Notes

- `LLMClient` and `EmbeddingClient` are provider-agnostic interfaces.
- This repo includes deterministic fallbacks and mock clients for local development/testing.
