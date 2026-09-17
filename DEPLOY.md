# Deploying

Same Render pattern as fame-battle-api: no database, just a Docker web
service.

| | where | what |
|---|---|---|
| API | Render (Docker web service) | FastAPI, ~900MB image (opencv + onnxruntime + rembg's matting deps) |

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
[render.yaml](render.yaml). No extra environment variables to set by hand
(unlike leaderboard-api, there's no `DATABASE_URL` to attach).

The image is large enough that the *first* build will take a while (Docker
layer with onnxruntime + scipy + scikit-image + rembg's dependency tree, plus
downloading and baking in both model weights). Subsequent deploys from an
unchanged `pyproject.toml` reuse that cached layer and are fast.

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
