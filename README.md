# Embedding API

An API for generating text embeddings using Microsoft's [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) model.

Supports queries and passages in 100+ languages.

---

## Running Locally

### Option 1: Poetry (Quick Start)

```bash
# Install dependencies
poetry install

# Start the server
poetry run uvicorn embedding_api.main:app --reload --port 8000
```

### Option 2: Docker Compose (With Monitoring)

```bash
# Start API + Prometheus + Grafana
docker-compose up --build

# Or run in background
docker-compose up -d
```

> **Note**: Monitoring (Prometheus + Grafana) is only available via Docker Compose.

---

## Using the API

### Swagger API Documentation

Once running, visit: http://localhost:8000/api/v1/docs

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "hello world2"]}'
```

Response:
```json
{
  "embeddings": [[...], [...]],
  "model": "intfloat/multilingual-e5-large"
}
```

---

## Monitoring

### Grafana Dashboards

Access Grafana at: http://localhost:3000

- **API Monitoring**: Request rates, latency, error rates
- **Model Monitoring**: Inference times, batch sizes, memory usage

Default credentials: `admin` / `admin`

### Prometheus

Metrics available at: http://localhost:9090

---

## Input Validation & Error Handling

### Input Validation

- **Request format**: Pydantic models validate all incoming requests
- **Required fields**: `texts` must be a non-empty list (max 32 items)
- **Task type**: Optional `task_type` must be either `"query"` or `"passage"`
- Invalid requests return `422 Unprocessable Entity` with clear error messages

### Error Handling

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `400` | Bad request |
| `422` | Validation error |
| `500` | Internal server error |
| `503` | Service unavailable (model not loaded) |

All errors return a JSON response with a `detail` field explaining the issue.

---

## Release strategy

The repository is set up to be compatible with [semantic versioning](https://semver.org/). Therefore, there is a release pipeline in the github workflows that runs on pushes to branches named something following the pattern: `[0-9]+.[0-9]+.x`.
The pipeline does the following:
1. Builds the image
2. Runs the test suite
3. Runs static code checks for security vulnerabilities
4. Push image to cloud container registry
5. Deploy to staging/test environment and notify team through teams/email
6. Perform simple tests to endpoints to check everything works
7. Deploy to production environment and notify team through teams/email

---

# Repository

## Commits and rulesets
The repository is set up to use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#specification) through branch protection rules.
The ruleset requires a pull request before merging to main. In a team, the PR should be reviewed and approved by at least 1 from the team who is not the author.

## Status checks
Status checks have also been set in place. In order to approve/merge a pull request, the continuous integration pipeline must pass. This includes linting and unit tests.
