# Deployment Guide

This project runs as **two independent pieces** so it can be fully hosted for
free — no need to keep your own computer running:

| Piece | What it is | Where it's hosted | Cost |
|-------|------------|-------------------|------|
| **Frontend** | `app/static/index.html` (the UI) | **GitHub Pages** (static) | Free |
| **Backend** | FastAPI API in `app/main.py` (`/laps`, `/analyze`) | **Render** (or any free Python host) | Free |

The frontend is just a web page. All the heavy work (reading your CSV, running
the comparison pipeline, downloading real F1 telemetry) happens in the backend.
The page calls the backend over HTTP, so the two can live on different servers.

```text
  Browser ──► GitHub Pages (index.html)
                   │  fetch(API_BASE + "/laps" | "/analyze")
                   ▼
            Render (FastAPI backend)  ──► FastF1 / analysis pipeline
```

---

## TL;DR (the 3 things you do once)

1. **Deploy the backend** to Render → you get a URL like
   `https://f1-ai-driver-coach-api.onrender.com`.
2. **Set that URL in the frontend**: open `app/static/index.html`, find
   `PROD_API_BASE` near the top of the `<script>`, and paste your backend URL.
3. **Push to `main`** → GitHub Pages redeploys the frontend automatically.

That's it. The site at `https://<you>.github.io/<repo>/` now works on its own.

---

## 1. Run it locally (development)

Everything on one machine, one origin — no API URL or CORS setup needed.

```bash
pip install -e ".[web]"          # installs FastAPI + the analysis pipeline
python -m app.main               # serves UI + API at http://127.0.0.1:8000
# or:  uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The backend serves the page itself, so the
frontend's API base is `""` (same origin) automatically — nothing to configure.

---

## 2. Deploy the backend to a free host (Render)

A [`render.yaml`](render.yaml) Blueprint is included, so deployment is mostly
clicks.

1. Push this repo to GitHub (if it isn't already).
2. Go to <https://render.com> and sign up (free).
3. **New + → Blueprint → select this repository.** Render reads `render.yaml`
   and creates a **free web service** that:
   - installs deps with `pip install -r requirements.txt`,
   - starts with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`,
   - health-checks `GET /health`.
4. Wait for the first build (a few minutes — it installs numpy/pandas/FastF1).
5. Copy your service URL, e.g. `https://f1-ai-driver-coach-api.onrender.com`.
   Test it: opening `…/health` should return `{"status":"ok"}`.

> ⚠️ **Root Directory must be blank (the repo root) — do NOT set it to `app/`.**
> `requirements.txt` is at the repo root, and the backend runs as the module
> `app.main` while importing `src/` and reading `tracks/`, `data/` and `cache/`,
> all of which only resolve from the repo root. The Blueprint pins this with
> `rootDir: .`. If you created the service **manually** and the build fails with
> `Could not open requirements file: requirements.txt`, the Root Directory is set
> to a subfolder — clear it under **Settings → Build & Deploy → Root Directory**
> (leave it empty) and redeploy.

**Free-tier note:** the service sleeps after ~15 min of inactivity. The next
request wakes it and can take ~30–60s — the UI shows a loading state, and the
error message tells you to wait if it times out.

> Prefer a different host? Any platform that runs a Python web process works.
> Use the same two commands:
> - build: `pip install -r requirements.txt`
> - start: `mkdir -p cache/fastf1 && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
>
> The `mkdir` is needed because `src/fastf1_loader.py` enables its FastF1 cache
> at import time. Render's start command (in `render.yaml`) already does this.

---

## 3. Point the frontend at your backend

The frontend needs to know your backend URL. Two ways:

### Option A — set it in the file (recommended for the public site)

Edit `app/static/index.html`, near the top of the `<script>` block:

```js
const PROD_API_BASE = "https://f1-ai-driver-coach-api.onrender.com";
```

Commit and push to `main`. GitHub Pages rebuilds, and the live site now calls
your backend. (When `PROD_API_BASE` is empty, a remotely-hosted page has no
backend and analysis shows a clear "set PROD_API_BASE" error.)

### Option B — override at runtime (no edit, great for testing)

Append `?api=` to the Pages URL:

```text
https://<you>.github.io/<repo>/?api=https://f1-ai-driver-coach-api.onrender.com
```

The value is remembered in the browser (localStorage) for next time. You can
also run `localStorage.setItem("apiBase", "https://…")` in the dev console, or
`localStorage.removeItem("apiBase")` to clear it.

**Resolution order:** `?api=` → stored value → `PROD_API_BASE` (only on a remote
host) → `""` (same origin, i.e. local dev).

---

## 4. GitHub Pages (frontend) — already automated

The workflow [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)
publishes the frontend on every push to `main` that touches `app/static/**`:

1. It copies `app/static/` into a `_site/` folder (so `index.html` is the site
   root), adds a `.nojekyll` marker, and deploys to Pages.
2. You can also run it manually: **Actions → "Deploy static site to GitHub
   Pages" → Run workflow.**

**One-time setup:** the workflow self-enables Pages (source = *GitHub Actions*).
If the first run doesn't enable it, set it once under **Settings → Pages →
Build and deployment → Source = GitHub Actions**. Do **not** pick "Deploy from a
branch".

After a green run, the URL appears in the workflow's `deploy` job and under
**Settings → Pages** (typically `https://<you>.github.io/<repo>/`).

**Custom domain:** add it under **Settings → Pages → Custom domain** (or drop a
`CNAME` file in `app/static/` so it's included in `_site/`). For a custom domain
you must also allow it on the backend — see CORS below.

---

## 5. CORS (cross-origin requests)

Because the page and the API are on different origins, the backend must allow
the page's origin. This is configured in `app/main.py` via `CORSMiddleware`.
**Allowed by default:**

- any `localhost` / `127.0.0.1` port (local dev), and
- any `https://<something>.github.io` site (GitHub Pages).

So a standard GitHub Pages deployment needs **no CORS configuration**.

Using a **custom domain** for the frontend? Add it to the backend's
`ALLOWED_ORIGINS` env var (comma-separated exact origins) — on Render:
**Service → Environment → Add** `ALLOWED_ORIGINS=https://your-domain.com`.
No cookies/credentials are used, so file uploads and analysis POSTs (and their
preflight `OPTIONS`) work cross-origin.

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Build fails: `Could not open requirements file: requirements.txt` | Render **Root Directory** is set to a subfolder (e.g. `app/`). `requirements.txt` is at the repo root — clear Root Directory (leave it blank) under Settings → Build & Deploy, or redeploy via the Blueprint (`rootDir: .`). |
| Build OK but start crashes (`ModuleNotFoundError: app`/`src`) | Same root-dir cause: run from the repo root so `app.main` and `src/` resolve. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. |
| "Set PROD_API_BASE…" error on the live site | Backend URL not configured — do step 3. |
| First analysis hangs ~30–60s then works | Render free service was asleep; it woke up. Normal. |
| "Couldn't reach the backend…" | Backend down or wrong URL. Check `…/health` returns `{"status":"ok"}`. |
| Browser console shows a **CORS** error | Frontend origin isn't allowed. On `*.github.io` it's automatic; for a custom domain set `ALLOWED_ORIGINS`. |
| **Mixed content** blocked | Your backend URL must be **https://** (Render URLs already are). |
| Page loads but is "static only" | That's the page with no backend configured — it always loads; analysis needs the backend (steps 2–3). |

---

## Summary of files

- `app/main.py` — FastAPI backend; CORS + reads `$HOST`/`$PORT`/`ALLOWED_ORIGINS`.
- `app/static/index.html` — frontend; `PROD_API_BASE` / `?api=` selects the backend.
- `requirements.txt` — backend dependencies for the host.
- `render.yaml` — Render free-tier Blueprint (build/start/health).
- `.github/workflows/deploy-pages.yml` — auto-publishes the frontend to Pages.
