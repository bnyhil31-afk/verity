# Meeting Notes — Project Alpha Sprint Review

**Date:** 2026-04-08
**Participants:** Jordan (PM), Alex (backend), Sam (frontend), Riley (QA)

## Agenda
1. Sprint 4 demo
2. Blocker review
3. Sprint 5 planning

## What Was Demoed
- Alex demoed the `GET /tasks` and `POST /tasks` endpoints with live Swagger UI
- Pagination works correctly; filtering by status is solid
- Soft delete behaviour confirmed — `deleted_at` is set, records remain in DB

## Decisions Made
- **Webhook retries** will use exponential backoff with a max of 5 attempts
  (Jordan approved, no customer-facing SLA impact for now)
- OpenAPI spec export will be automated as part of the CI pipeline, not manual
- PostgreSQL migration will happen in the staging environment on April 15th;
  data team will review schemas by April 13th

## Action Items
- [ ] Alex: implement webhook retry logic (by April 11th) — see project_alpha issue #34
- [ ] Sam: integrate new camelCase field names from latest API changes
- [ ] Riley: write test cases for PATCH edge cases (null fields, concurrent updates)
- [ ] Jordan: confirm staging deploy window with ops team

## Notes
Sam raised a concern about response latency on the paginated list endpoint when
there are more than 10,000 tasks. Alex will add a database index on `status` and
`assignee_id` before the staging deploy.
