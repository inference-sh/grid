# Mirage Provider

Talking-head video generation, styled captions, and avatar creation. Formerly Captions —
the company rebranded to Mirage in September 2025, which is why two API hosts are live.

## Apps

| App | Endpoint | Host | Category | Description |
|-----|----------|------|----------|-------------|
| [video-1](video-1/) | `POST /v1/videos` | mirage.app | video | Talking-head video from a portrait image + audio track |
| [video-captions](video-captions/) | `POST /v1/videos/captions` | mirage.app | video | Burn styled animated captions onto a vertical video |
| [ai-creator](ai-creator/) | `POST /creator/submit` | captions.ai | video | Script → spokesperson video with an AI Creator or your AI Twin |
| [ai-ads](ai-ads/) | `POST /ads/submit` | captions.ai | video | Script + product media → UGC-style ad video |
| [ai-twin](ai-twin/) | `POST /twin/create` | captions.ai | other | Build a reusable avatar from a calibration video + stills |

## Two hosts, one key

Both API generations are live and both authenticate with the same `x-api-key` header,
supplied by the `CAPTIONS_KEY` secret.

- **`https://api.mirage.app/v1`** — the current platform. Multipart submit, `GET /videos/{id}`
  polling, result served by redirect from `/videos/{id}/content`. Documented per-second
  pricing, so this half is GA.
- **`https://api.captions.ai/api`** — the original Captions API. JSON submit returning an
  `operationId`, polled by POSTing that ID back to a `/poll` or `/status` endpoint, result
  handed over as a plain URL.

Keys are issued from the [Mirage platform dashboard](https://platform.mirage.app/). Whether one
key covers both hosts is not documented; `_raise_for_error` in `mirage_helper.py` calls this out
explicitly on a 401 so a cross-host key mismatch is not mistaken for an entitlement problem.

## How the apps compose

```
mirage/ai-twin  ──(twin_name)──>  mirage/ai-creator   (script → video, your likeness)
                              └>  mirage/ai-ads       (script + product media → ad)
```

`ai-twin` returns the twin's full name; pass it as `creator_name` to either of the
other two. Nothing lists twins, so that returned name is the only record of it.

## Multi-tenancy on a shared key

Every caller shares one `CAPTIONS_KEY`, and the provider applies no per-caller
ownership check. Neither API has a tag field or a filter parameter on any
endpoint — verified against the full OpenAPI surface — so resources cannot be
scoped server-side. Three consequences are designed in:

- **No listing.** `/creator/list` and `/ads/list-creators` take no arguments and
  return every creator on the account, including other tenants' twins. Since the
  result cannot be filtered server-side, those functions are not exposed at all.
  `list_templates` on `video-captions` is the one exception: caption templates are
  provider-global stock styles, published in Mirage's own docs, and contain no
  tenant data.
- **No id inputs.** The captions endpoint also accepts a bare `video_id`, which
  would let any caller caption another tenant's video and receive a copy. Not
  exposed; uploads only.
- **Unguessable twin names.** `creator_name` is a free-form string, so a twin
  named `spokesperson` could be performed by anyone who guesses that word.
  `scoped_name()` gives every twin a random component. `team_id`/`task_id` are
  folded in for traceability when the engine supplies them, but the random part
  is what does the security work — see the note in `mirage_helper.py`.

## Choosing between the talking-head apps

- **`video-1`** — you have (or generate) the audio. Best fidelity, any voice, any language your
  audio is in. Billed per second of output.
- **`ai-creator`** — you have only text. The voice is synthesized and the performer is one of
  Captions' stock creators or an AI Twin you built. Up to 4K. You must already know the
  creator name — nothing enumerates them.

## Constraints worth knowing before wiring these up

- **Video generation is rate limited to 2 requests/min** per organization. Captions is 100/min.
- `video-captions` input must be **9:16 vertical, ≤50 MB, ≤5 minutes**, MP4 or MOV.
- `ai-ads` and `ai-twin` fetch media **by URL** and cannot accept a local upload — inputs must
  resolve to a public URL. `video-1` and `video-captions` upload directly, so local files work.
- `ai-creator` and `ai-ads` scripts are capped at **800 characters**.
- Webhooks exist on the legacy host but are gated behind an Enterprise contract, so all five
  apps poll instead.

## Pricing

See [pricing.md](pricing.md).

## Shared code

`mirage_helper.py` lives at the provider root and is **symlinked** into each app. Check before
editing a copy:

```bash
for d in */; do printf "%-18s " "$d"; [ -L "$d/mirage_helper.py" ] && echo LINK || echo COPY; done
```
