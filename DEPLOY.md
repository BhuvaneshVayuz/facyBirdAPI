# Deploying

Same Render pattern as the other services: a Docker web service, plus an
*optional* Postgres connection for the cutout cache (the same Neon database
leaderboard-api uses). Skip the database entirely and the service still
works -- it just recomputes every cutout.

| | where | what |
|---|---|---|
| API | Render (Docker web service) | FastAPI, ~451MB image (opencv drives both models directly via cv2.dnn -- see README's "Render Free tier" section for why onnxruntime/rembg aren't dependencies) |

---

## 1. Put it in git

```bash
git init
git add .
git commit -m "facy-bird-api"
git branch -M main
git remote add origin https://github.com/<you>/facy-bird-api.git
git push -u origin main
```

Confirm `.env` isn't staged -- it's gitignored already, same as every other
repo in this project.

## 2. API on Render

Dashboard -> **New** -> **Blueprint** -> select the repo -> reads
[render.yaml](render.yaml).

One optional variable to set by hand afterwards, the same way
leaderboard-api's is set (Neon connection strings can't be auto-injected by
a Render blueprint): **Environment** -> add `DATABASE_URL` = the *same* Neon
connection string leaderboard-api uses. This service creates its own
`face_cutouts` table there and touches nothing else -- no foreign key to
`users`, so the two services stay independent and deploy order doesn't
matter.

Leaving it unset is a supported configuration, not a broken one: the cutout
cache logs that it's disabled and every request recomputes.

The *first* build takes a little while (installing opencv, then downloading
and baking in both model weights). Subsequent deploys from an unchanged
`pyproject.toml` reuse that cached layer and are fast.

Then check it:

```bash
curl https://<your-service>.onrender.com/health
```

Expect `{"ok": true}`.

## 3. Point flappy-bird-web at it

`facyBirdApiBase` (an env var at build time or a prop passed to
`open()`/`mount()` -- see that repo's README for which it uses), same
`VITE_*`-at-build-not-runtime caveat as every other `VITE_*` value in this
project.

## 4. Close the CORS default

Same posture as every other service here: `CORS_ALLOW_ORIGINS` ships as `*`
so nothing blocks during initial setup. Narrow it to the actual origins
flappy-bird-web gets embedded on once known:

```
CORS_ALLOW_ORIGINS=https://your-host-app.com
```

## 5. Free-tier cold starts

Render's free web services sleep after inactivity. Because both model
weights are baked into the image (not downloaded on first request), a cold
start here costs model *load* time, not model *download* time -- noticeably
better than it would be otherwise, but still a real few-second wait on the
first "Facy Bird" request after the service has been asleep. The game's
loading state ("Preparing your bird...") is what carries the player through
that, same idea as any other cold-start-prone free-tier service.

---

## Running locally

```bash
uvicorn src.api:app --reload
```

## Testing the container before you push

```bash
docker build -t facy-bird-api .
docker run --rm -p 8000:8000 facy-bird-api
curl -X POST localhost:8000/face-cutout \
  -H "Content-Type: application/json" \
  -d '{"photo_url": "https://example.com/a-real-single-face-photo.jpg"}' \
  -o cutout.png
```
