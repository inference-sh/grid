---
name: using-oauth-integrations
description: Use OAuth integrations in inference.sh apps. Use when accessing Google Sheets, Drive, or other OAuth services on behalf of users.
---

# Using OAuth Integrations

Access external services (Google Sheets, Drive) on behalf of users through OAuth.

## Declaring Integrations

In `inf.yml`:

```yaml
integrations:
  - key: google.sheets
    description: Read/write Google Sheets
    optional: false
```

## Available Integrations

```bash
infsh integrations list
```

| Key | Description |
|-----|-------------|
| `google.sheets` | Read/write Sheets |
| `google.sheets.readonly` | Read-only Sheets |
| `google.drive` | Google Drive files |
| `google.sa` | Service account |

## Accessing Credentials

```python
import os, json

creds_json = os.environ.get("GOOGLE_OAUTH_CREDENTIALS")
if creds_json:
    credentials = json.loads(creds_json)
```

## Service Account

```python
from google.oauth2 import service_account

sa_json = os.environ.get("GOOGLE_SA_CREDENTIALS")
if sa_json:
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(sa_json)
    )
```

📖 **Full docs**: [inference.sh/docs/extend/integrations](https://inference.sh/docs/extend/integrations)
