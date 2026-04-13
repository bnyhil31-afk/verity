# Project Alpha — Web Service Notes

## Overview
Building a lightweight REST API for internal task tracking. Stack: FastAPI + SQLite
for local dev, PostgreSQL for staging and production.

## Technical Decisions
- Chose FastAPI over Flask for automatic OpenAPI docs and async support
- SQLAlchemy 2.x async sessions for database access (avoids blocking I/O)
- JWT auth with a 24-hour access token and 7-day refresh token
- All endpoints return camelCase JSON (frontend requirement)

## Current Architecture
- `GET /tasks` — paginated list, supports ?status=open|closed&assignee=<id>
- `POST /tasks` — create task, triggers notification webhook if assignee is set
- `PATCH /tasks/{id}` — partial update, audit-logged
- `DELETE /tasks/{id}` — soft delete only, sets deleted_at timestamp

## Blockers
- The webhook retry logic is not implemented yet (issue #34)
- PostgreSQL migration scripts need review before staging deploy
- Frontend team needs the OpenAPI spec exported and checked in

## Next Steps
1. Implement exponential-backoff retry for webhooks (assign to self)
2. Write integration tests for the PATCH endpoint edge cases
3. Schedule schema review with the data team for the week of the 21st
4. Export OpenAPI spec to `docs/api/openapi.json`

## Notes
The SQLite dev setup runs fine with `uvicorn app.main:app --reload`.
Remember to set `DATABASE_URL` in `.env` before running locally.
