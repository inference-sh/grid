# Pricing

What the vendor charges for each function this app exposes, so the app's price can be set from it.

- source_url: <the vendor's pricing page>
- checked: <YYYY-MM-DD>
- confidence: <high | medium | low: how sure the rates below are, and why>

| function | upstream rate | unit | notes |
|----------|---------------|------|-------|
| list_task_tracking_integrations | | request | GET /task_tracking/integrations |
| link_task_to_issue_group | | request | POST /task_tracking/linkTaskToIssueGroup |
| map_repos_to_project | | request | POST /task_tracking/mapCodeReposToProjects |
| map_teams_to_project | | request | POST /task_tracking/mapTeamsToProjects |
| get_project_mapping | | request | GET /task_tracking/projectMapping |
| get_projects | | request | GET /task_tracking/projects |
| get_tasks | | request | GET /task_tracking/projects/{project_id}/tasks |
| unlink_task_from_issue_group | | request | POST /task_tracking/unlinkTaskFromIssueGroup |

A metered function reports what it consumed in `output_meta` (for example
`OutputMeta(outputs=[RawMeta(cost=...)])`), and the app's price is set from that.
A free function reports `OutputMeta(inputs=[], outputs=[])`.
