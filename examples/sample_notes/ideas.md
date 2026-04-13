# Ideas & Brainstorm

## API improvements
- Add bulk task creation endpoint (`POST /tasks/bulk`) — would unblock the import tool
- GraphQL layer on top of the REST API? Maybe later, keep REST for now
- Rate limiting per API key using a token bucket algorithm

## Developer experience
- Pre-commit hooks for linting and type checking would catch issues earlier
- Generate a local dev certificate so the frontend can test HTTPS end-to-end
- Document the webhook payload format in the OpenAPI spec (currently missing)

## Future connectors
- Slack integration: create tasks directly from Slack messages
- GitHub integration: auto-create tasks from issues tagged `project-alpha`
- Calendar integration: pull meeting notes into the knowledge base automatically

## Performance
- The paginated list endpoint will need indexing on `status` and `assignee_id`
  once data volume grows (discussed in last Project Alpha sprint review)
- Consider caching the OpenAPI spec generation — it's expensive on every cold start
