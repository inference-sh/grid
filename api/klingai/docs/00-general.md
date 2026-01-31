# Kling AI API - General Information

## API Domain

```
https://api-singapore.klingai.com
```

## Authentication

### Step 1: Obtain Credentials
Get your `AccessKey` and `SecretKey` from the Kling AI platform.

### Step 2: Generate JWT Token

```python
import time
import jwt

ak = "your_access_key"
sk = "your_secret_key"

def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,  # Valid for 30 minutes
        "nbf": int(time.time()) - 5       # Starts 5 seconds ago
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token
```

### Step 3: Use in Request Header

```
Authorization: Bearer <API_TOKEN>
```

Note: There must be a space between "Bearer" and the token.

---

## Error Codes

| HTTP | Code | Definition | Solution |
|------|------|------------|----------|
| 200 | 0 | Success | - |
| 401 | 1000 | Authentication failed | Check Authorization |
| 401 | 1001 | Authorization empty | Fill Authorization header |
| 401 | 1002 | Authorization invalid | Check token format |
| 401 | 1003 | Token not yet valid | Wait or regenerate |
| 401 | 1004 | Token expired | Regenerate token |
| 429 | 1100 | Account exception | Verify account config |
| 429 | 1101 | Account in arrears | Recharge account |
| 429 | 1102 | Resource pack depleted | Purchase more resources |
| 403 | 1103 | Unauthorized access | Verify permissions |
| 400 | 1200 | Invalid parameters | Check request params |
| 400 | 1201 | Invalid param value | See message field |
| 404 | 1202 | Invalid method | Check API docs |
| 404 | 1203 | Resource not found | Check model name |
| 400 | 1300 | Platform policy triggered | Review content |
| 400 | 1301 | Content security policy | Modify input content |
| 429 | 1302 | Rate limit exceeded | Reduce frequency |
| 429 | 1303 | Concurrency limit | Reduce concurrent tasks |
| 429 | 1304 | IP whitelist policy | Contact support |
| 500 | 5000 | Server error | Retry later |
| 503 | 5001 | Server unavailable | Retry later |
| 504 | 5002 | Server timeout | Retry later |

---

## Concurrency Rules

### What is Concurrency?
Maximum parallel generation tasks. Determined by resource package.

### Key Rules
- Applied at **account level**
- Calculated **per resource pack type** (video/image/virtual try-on)
- All API keys under same account share quota
- Task occupies concurrency from `submitted` until completion
- Released immediately after task ends

### Concurrency Consumption
| Task Type | Consumption |
|-----------|-------------|
| Video generation | 1 per task |
| Virtual try-on | 1 per task |
| Image generation | = `n` parameter (e.g., n=9 uses 9 slots) |

### Over-limit Error
```json
{
    "code": 1303,
    "message": "parallel task over resource pack limit",
    "request_id": "uuid"
}
```

### Handling Strategy
1. **Exponential Backoff**: Initial delay ≥ 1 second
2. **Queue Management**: Control submission rate

---

## Task Statuses

| Status | Description |
|--------|-------------|
| `submitted` | Task accepted, queued |
| `processing` | Being generated |
| `succeed` | Completed successfully |
| `failed` | Failed (check `task_status_msg`) |

---

## File Requirements

### Images
| Requirement | Value |
|-------------|-------|
| Formats | `.jpg`, `.jpeg`, `.png` |
| Max size | 10MB |
| Min dimensions | 300px (width and height) |
| Aspect ratio | 1:2.5 to 2.5:1 |

### Base64 Format
**Correct:**
```
iVBORw0KGgoAAAANSUhEUgAAAAUA...
```

**Wrong (has prefix):**
```
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...
```

### Videos (for Omni-Video)
| Requirement | Value |
|-------------|-------|
| Formats | `.mp4`, `.mov` |
| Duration | 3-10 seconds |
| Resolution | 720px to 2160px |
| Frame rate | 24-60 fps (output: 24fps) |
| Max size | 200MB |

---

## Content Retention

**Generated images/videos are deleted after 30 days.** Save them promptly.

---

## Callback Protocol

Set `callback_url` when creating tasks to receive webhooks:

```json
{
  "code": 0,
  "message": "string",
  "request_id": "string",
  "data": {
    "task_id": "string",
    "task_status": "succeed",
    "task_result": { ... },
    "created_at": 1722769557708,
    "updated_at": 1722769557708
  }
}
```
