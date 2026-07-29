# Mirage Provider

Talking-head video generation and styled captions. Formerly Captions — the company
rebranded to Mirage in September 2025, which is why two API hosts appear here. Only
`api.mirage.app` is reachable; `api.captions.ai` no longer serves traffic.

## Apps

| App | Endpoint | Host | Category | Description |
|-----|----------|------|----------|-------------|
| [video-1](video-1/) | `POST /v1/videos` | mirage.app | video | Talking-head video from a portrait image + audio track |
| [video-captions](video-captions/) | `POST /v1/videos/captions` | mirage.app | video | Burn styled animated captions onto a vertical video (67 styles) |
| [text-overlays](text-overlays/) | `POST /v1/meta/text_overlays` | mirage.app | video | Render up to 4 static text variants onto one video |
| [ai-creator](ai-creator/) | `POST /creator/submit` | captions.ai | video | **DARK** — script → spokesperson video |
| [ai-ads](ai-ads/) | `POST /ads/submit` | captions.ai | video | **DARK** — script + product media → ad video |
| [ai-twin](ai-twin/) | `POST /twin/create` | captions.ai | other | **DARK** — avatar from a calibration video + stills |

The three `captions.ai` apps are deployed but non-functional: their host is down. Code is
correct against the API as documented, so they would work if it returns. See below.

## Two hosts, one dead

Both generations share the `x-api-key` header, supplied by the `CAPTIONS_KEY` secret. Keys
come from the [Mirage platform dashboard](https://platform.mirage.app/).

- **`https://api.mirage.app/v1`** — current and working. Multipart submit, `GET /videos/{id}`
  polling, result by redirect from `/videos/{id}/content`. Verified end to end.
- **`https://api.captions.ai/api`** — legacy, **no healthy backend**. Every endpoint including
  `/` returns 502, with an invalid key, from multiple networks, sustained over ~40 minutes.

The 502 is a load balancer with nothing behind it, not an auth or routing error:

```
api.captions.ai   HTTP/2 502   (no server: header, no via: header)
api.mirage.app    HTTP/2 405   server: uvicorn, via: 1.1 google
```

DNS also points it at a dev cluster — `api.captions.ai` → `34.54.177.12`, aliased
`k1.captions-dev.xyz`. The legacy doc pages (`ai-creator/submit`, `ai-ads/submit`,
`ai-twin/create`, `webhooks`) were removed during a `help.mirage.app` → `captions.ai/help`
migration; the current api-reference index lists only `videos/*`.

Checked and ruled out: the legacy paths did **not** move to the new host. Every variant
(`/api/creator/list`, `/v1/creator/list`, `/v1/twin/create`, …) returns **404 on
api.mirage.app** — routing works, the routes don't exist — versus 502 on the old host.

Whether this is a decommission or a long outage can't be established from outside. What is
certain: nothing on our side fixes it.

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
- `video-1`'s input image must be **9:16 or 16:9**. A square image is rejected with
  `Image must have a 9:16 or 16:9 aspect ratio`. This is undocumented upstream — found by
  hitting it, so don't expect the docs to mention it.
- `video-captions` input must be **9:16 vertical, ≤50 MB, ≤5 minutes**, MP4 or MOV. Its audio
  needs actual speech; a tone or silence fails the job with no error detail.
- `ai-ads` and `ai-twin` fetch media **by URL** and cannot accept a local upload — inputs must
  resolve to a public URL. `video-1` and `video-captions` upload directly, so local files work.
- `ai-creator` and `ai-ads` scripts are capped at **800 characters**.
- Webhooks exist on the legacy host but are gated behind an Enterprise contract, so all five
  apps poll instead.

## Undocumented endpoints on api.mirage.app

`https://api.mirage.app/openapi.json` is served by the API host itself and lists more
than the published docs do. Two public endpoints have no app yet:

- `POST /v1/audio/text-to-speech/{voice_id}` — Mirage's own TTS (`model: mirage-audio-1`),
  pairs with `video-1` which needs a speech track. This is the remaining gap: `voice_id` is a bare
  string with no enum, no example, and **no endpoint anywhere enumerates voices**. A valid
  id has to come from Mirage — do not guess one.

`text_overlays` is now built out as `mirage/text-overlays`. Neither endpoint has published
pricing. Prefer that live openapi.json over the help-centre pages:
the docs lag it, and the legacy pages were deleted outright rather than updated.

## Captions vs text overlays

Two different jobs, hence two endpoints:

| | `video-captions` | `text-overlays` |
|---|---|---|
| Text source | transcribed from the audio | you supply it |
| Timing | animated, per-word to speech | static |
| Outputs | 1 | up to 4 from one upload |
| Styling | 67 named templates | raw font/size/colour per variant |
| Pricing | $0.15/min of input | not published |

`text-overlays` fans out: the job carries a `results[]` with a per-variant `COMPLETE`/`FAILED`
status, so a job can finish `COMPLETE` with individual variants failed. The app reports each
variant instead of collapsing them, and only raises if every one failed. Its `fonts`/`sizes`/
`colors` lists are matched to `texts` **by position**, so the app rejects mismatched lengths
rather than letting styles silently shift onto the wrong variant.

## Caption styles

`video-captions` bakes all 67 style names into a dropdown (fetched 2026-07-29) so callers
pick `caption_style: "Neon"` rather than an opaque `ctpl_` string. Mirage adds styles over
time, so the field is paired with a raw `caption_template_id` override, and the
`list_templates` function returns the live set with preview videos. To refresh the baked
list, run `list_templates` with `limit: 100` and regenerate the map.

## Pricing

See [pricing.md](pricing.md).

## Shared code

`mirage_helper.py` lives at the provider root and is **symlinked** into each app. Check before
editing a copy:

```bash
for d in */; do printf "%-18s " "$d"; [ -L "$d/mirage_helper.py" ] && echo LINK || echo COPY; done
```
