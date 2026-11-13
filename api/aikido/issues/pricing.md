# Pricing

What the vendor charges for each function this app exposes, so the app's price can be set from it.

- source_url: <the vendor's pricing page>
- checked: <YYYY-MM-DD>
- confidence: <high | medium | low: how sure the rates below are, and why>

| function | upstream rate | unit | notes |
|----------|---------------|------|-------|
| list_open_issue_groups | | request | GET /open-issue-groups |
| get_issue_counts | | request | GET /issues/counts |
| get_issue_cve_exploitability_bulk | | request | GET /issues/cveExploitability/bulk |
| get_issue_details_bulk | | request | GET /issues/detail/bulk |
| export_issues | | request | GET /issues/export |
| get_issue_group_details | | request | GET /issues/groups/{issue_group_id} |
| ignore_issue_group | | request | PUT /issues/groups/{issue_group_id}/ignore |
| list_notes_for_issue_group | | request | GET /issues/groups/{issue_group_id}/notes |
| add_note_to_issue_group | | request | POST /issues/groups/{issue_group_id}/notes |
| adjust_group_severity | | request | POST /issues/groups/{issue_group_id}/severity/adjust |
| snooze_issue_group | | request | PUT /issues/groups/{issue_group_id}/snooze |
| get_issue_group_tasks | | request | GET /issues/groups/{issue_group_id}/tasks |
| unignore_issue_group | | request | PUT /issues/groups/{issue_group_id}/unignore |
| unsnooze_issue_group | | request | PUT /issues/groups/{issue_group_id}/unsnooze |
| get_issue_detail | | request | GET /issues/{issue_id} |
| get_issue_cve_exploitability | | request | GET /issues/{issue_id}/cveExploitability |
| ignore_issue | | request | PUT /issues/{issue_id}/ignore |
| get_issue_reachability | | request | GET /issues/{issue_id}/reachability |
| adjust_severity | | request | POST /issues/{issue_id}/severity/adjust |
| snooze_issue | | request | PUT /issues/{issue_id}/snooze |
| solve_issue | | request | PUT /issues/{issue_id}/solve |
| unignore_issue | | request | PUT /issues/{issue_id}/unignore |
| unsnooze_issue | | request | PUT /issues/{issue_id}/unsnooze |

A metered function reports what it consumed in `output_meta` (for example
`OutputMeta(outputs=[RawMeta(cost=...)])`), and the app's price is set from that.
A free function reports `OutputMeta(inputs=[], outputs=[])`.
