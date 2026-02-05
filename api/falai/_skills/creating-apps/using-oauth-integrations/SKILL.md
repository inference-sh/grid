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

  - key: google.drive
    description: Access to Google Drive files
    optional: true
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `key` | string | Integration identifier |
| `description` | string | Shown to users |
| `optional` | boolean | If false, app won't run without it |

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

### OAuth Integrations

```python
import os, json

class App(BaseApp):
    async def setup(self, config):
        creds_json = os.environ.get("GOOGLE_OAUTH_CREDENTIALS")
        if creds_json:
            self.credentials = json.loads(creds_json)
```

### Service Account

```python
from google.oauth2 import service_account

class App(BaseApp):
    async def setup(self, config):
        sa_json = os.environ.get("GOOGLE_SA_CREDENTIALS")
        if sa_json:
            self.credentials = service_account.Credentials.from_service_account_info(
                json.loads(sa_json)
            )
```

## Secrets vs Integrations

| Feature | Secrets | Integrations |
|---------|---------|--------------|
| User provides | Raw value (API key) | OAuth authorization |
| Refresh | Manual | Automatic |
| Scope control | None | Fine-grained |
| Best for | API keys | OAuth services |

## Best Practices

1. **Request minimal scopes** - use `readonly` if you only read
2. **Clear descriptions** - explain why access is needed
3. **Handle missing gracefully** - check if optional integrations exist
