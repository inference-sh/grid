# X API Apps

Apps for interacting with the X (Twitter) API using the official `xdk` Python SDK.

## SDK Usage

```python
from xdk import Client

# Initialize with OAuth 2.0 access token
client = Client(access_token=os.environ.get("X_ACCESS_TOKEN"))

# Posts
response = client.posts.create(body={"text": "Hello world"})
response = client.posts.get(id="123456789")
response = client.posts.delete(id="123456789")
response = client.posts.search_recent(query="api", max_results=10)

# Media upload (chunked)
init = client.media.initialize_upload(body={"total_bytes": size, "media_type": "image/jpeg", "media_category": "tweet_image"})
media_id = init.data["id"]
client.media.append_upload(id=media_id, body={"media": base64_chunk, "segment_index": 0})
client.media.finalize_upload(id=media_id)
client.media.get_upload_status(media_id=media_id)

# Users
response = client.users.get(id="123456789")
response = client.users.get_by_username(username="elonmusk")
response = client.users.follow(target_user_id="123456789")
response = client.users.unfollow(target_user_id="123456789")
response = client.users.get_followers(id="123456789")
response = client.users.get_following(id="123456789")

# Likes
response = client.likes.create(tweet_id="123456789")
response = client.likes.delete(tweet_id="123456789")

# Retweets/Reposts
response = client.reposts.create(tweet_id="123456789")
response = client.reposts.delete(tweet_id="123456789")

# Direct Messages
response = client.direct_messages.create(participant_id="123456789", body={"text": "Hello"})
response = client.direct_messages.get_events()

# Bookmarks
response = client.bookmarks.create(tweet_id="123456789")
response = client.bookmarks.delete(tweet_id="123456789")
```

## Response Format

All responses have a `.data` dict containing the result:

```python
response = client.posts.create(body={"text": "Hello"})
tweet_id = response.data["id"]
```

## API Pricing

| Category | Operation | Cost |
|----------|-----------|------|
| **Posts: Read** | Fetch posts | $0.005/resource |
| **User: Read** | Fetch user profiles | $0.010/resource |
| **DM Event: Read** | Fetch DM events | $0.010/resource |
| **Content: Create** | Create posts/media | $0.010/request |
| **DM Interaction: Create** | Send DMs | $0.015/request |
| **User Interaction: Create** | Follow, like, retweet | $0.015/request |
| **Interaction: Delete** | Unfollow, unlike, unretweet | $0.010/request |
| **Content: Manage** | Delete/hide posts | $0.005/request |
| **List: Read** | Fetch lists | $0.005/resource |
| **List: Create** | Create lists | $0.010/request |
| **List: Manage** | Update/delete lists | $0.005/request |
| **Space: Read** | Fetch spaces | $0.005/resource |
| **Community: Read** | Fetch communities | $0.005/resource |
| **Note: Read** | Fetch notes | $0.005/resource |
| **Following/Followers: Read** | Fetch social graph | $0.010/resource |
| **Media: Read** | Fetch media | $0.005/resource |
| **Analytics: Read** | Fetch analytics | $0.005/resource |
| **Bookmark** | Create bookmarks | $0.005/request |
| **Media Metadata** | Create/delete metadata | $0.005/request |
| **Privacy: Update** | Update privacy settings | $0.010/request |
| **Mute: Delete** | Delete mutes | $0.005/request |
| **Counts: Recent** | Fetch recent counts | $0.005/request |
| **Counts: All** | Fetch all counts | $0.010/request |
| **Trend: Read** | Fetch trends | $0.010/resource |

## Apps

### Implemented

| App | Description | Integration Key |
|-----|-------------|-----------------|
| `post-tweet` | Create posts with text and media | `x.tweet.write` |

### Planned

| App | Description | API Category |
|-----|-------------|--------------|
| `post-create` | Create posts (cleaner name) | Content: Create |
| `post-get` | Get post by ID | Posts: Read |
| `post-delete` | Delete a post | Content: Manage |
| `post-like` | Like a post | User Interaction: Create |
| `post-retweet` | Retweet a post | User Interaction: Create |
| `user-get` | Get user profile | User: Read |
| `user-follow` | Follow a user | User Interaction: Create |
| `dm-send` | Send direct message | DM Interaction: Create |

### Future

| App | Description | API Category |
|-----|-------------|--------------|
| `post-unlike` | Unlike a post | Interaction: Delete |
| `post-unretweet` | Undo retweet | Interaction: Delete |
| `user-unfollow` | Unfollow a user | Interaction: Delete |
| `posts-search` | Search posts | Posts: Read |
| `timeline-get` | Get home timeline | Posts: Read |
| `followers-list` | Get user's followers | Following/Followers: Read |
| `following-list` | Get user's following | Following/Followers: Read |
| `dm-list` | List DM conversations | DM Event: Read |
| `bookmark-add` | Bookmark a post | Bookmark |
| `bookmark-remove` | Remove bookmark | Bookmark |
| `list-create` | Create a list | List: Create |
| `list-get` | Get list by ID | List: Read |
| `list-members` | Get list members | List: Read |

## App Structure

Each app follows this structure:

```
app-name/
├── inf.yml           # App config (name, description, integrations, resources)
├── inference.py      # Main app logic
├── requirements.txt  # Python dependencies
├── packages.txt      # System packages (optional)
└── skills/           # Claude Code skills for development
```

### inf.yml Template

```yaml
name: app-name
description: Description of what the app does
metadata: {}
category: social
images:
    card: "https://cloud.inference.sh/app/files/u/4mg21r6ta37mpaz6ktzwtt8krr/01kg60feh39gmsptqz5gh2vnbh.jpeg"
    thumbnail: "https://cloud.inference.sh/app/files/u/4mg21r6ta37mpaz6ktzwtt8krr/01kg60fek57rq9c99mrrrs95t5.jpeg"
    banner: "https://cloud.inference.sh/app/files/u/4mg21r6ta37mpaz6ktzwtt8krr/01kg60fdwr4j81gkgas45fhd25.jpeg"
env: {}
kernel: python-3.11

integrations:
  - key: x.scope.permission
    description: What this integration allows

resources:
    gpu:
        count: 0
        vram: 0
        type: none
    ram: 2000
```

### Integration Keys

| Key | Description |
|-----|-------------|
| `x.tweet.write` | Create/delete posts |
| `x.tweet.read` | Read posts |
| `x.tweet.moderate` | Hide/unhide replies |
| `x.media.write` | Upload media |
| `x.user.read` | Read user profiles |
| `x.follow.write` | Follow/unfollow users |
| `x.follow.read` | Read followers/following |
| `x.like.write` | Like/unlike posts |
| `x.like.read` | Read likes |
| `x.dm.write` | Send DMs |
| `x.dm.read` | Read DMs |
| `x.bookmark.write` | Create/delete bookmarks |
| `x.bookmark.read` | Read bookmarks |
| `x.list.write` | Create/manage lists |
| `x.list.read` | Read lists |
| `x.block.write` | Block/unblock accounts |
| `x.block.read` | Read blocked accounts |
| `x.mute.write` | Mute/unmute accounts |
| `x.mute.read` | Read muted accounts |
| `x.space.read` | Read Spaces info |

### inference.py Template

```python
import os
from xdk import Client
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput
from pydantic import Field
from typing import Optional

class AppInput(BaseAppInput):
    """Input schema."""
    param: str = Field(description="Description")

class AppOutput(BaseAppOutput):
    """Output schema."""
    result: str = Field(description="Description")

class App(BaseApp):
    client: Client = None

    async def setup(self):
        access_token = os.environ.get("X_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("X_ACCESS_TOKEN not found")
        self.client = Client(access_token=access_token)

    async def run(self, input_data: AppInput) -> AppOutput:
        try:
            # API call here
            response = self.client.posts.get(id=input_data.param)
            return AppOutput(result=response.data["id"])
        except Exception as e:
            raise ValueError(f"X API error: {e}")

    async def unload(self):
        self.client = None
```

### requirements.txt

```
pydantic >= 2.0.0
inferencesh
xdk
```

## Media Upload

For uploading images/videos, use chunked upload:

1. **Initialize**: Get media_id
2. **Append**: Upload base64-encoded chunks (1MB each)
3. **Finalize**: Complete upload
4. **Poll status**: For videos, wait for processing

### Size Limits

| Type | Max Size |
|------|----------|
| Image | 5 MB |
| GIF | 15 MB |
| Video | 512 MB |

### Media Categories

| Content Type | Category |
|--------------|----------|
| `image/*` | `tweet_image` |
| `image/gif` | `tweet_gif` |
| `video/*` | `tweet_video` |

## Error Handling

Common error patterns:

```python
try:
    response = self.client.posts.create(body=payload)
except Exception as e:
    error_msg = str(e).lower()
    if "duplicate" in error_msg:
        raise ValueError("Duplicate content")
    elif "rate limit" in error_msg:
        raise ValueError("Rate limit exceeded")
    elif "unauthorized" in error_msg or "403" in error_msg:
        raise ValueError(f"Authorization failed: {e}")
    else:
        raise ValueError(f"X API error: {e}")
```

## Resources

- [X API Documentation](https://docs.x.com)
- [X Python SDK](https://docs.x.com/xdks/python/quickstart)
- [X Developer Portal](https://developer.x.com)
