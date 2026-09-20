# Toxicity Detection (Perspective API)

Batch scoring of 4chan /pol/ posts with the Google Perspective API for Severe Toxicity and Identity Attack attributes.

## Scripts

- `query_and_analyze_batch.py` — query posts from the database, score with Perspective API, and write results back
- `config.py` — shared configuration (DB credentials, API key, target keywords)

## Configuration

Set credentials via environment variables (no secrets stored in code):

| Variable | Default | Description |
|----------|---------|-------------|
| `PERSPECTIVE_API_KEY` | — | Google Perspective API key (required) |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `postgres` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | — | Database password |

## Usage

```bash
# Score posts mentioning india/indian (sample up to 100K)
python query_and_analyze_batch.py \
  --keywords "india" "indian" \
  --batch-size 500 \
  --max-total 100000
```

Results are written to a `perspective_results` table in the database.
