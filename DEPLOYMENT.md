# Deployment — GitHub Pages

The frontend (`app/static/index.html`) is published to GitHub Pages as a static
site by the workflow [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml).

## How it works

1. On every push to `main` that touches `app/static/**` (or the workflow file),
   the workflow copies `app/static/` into a `_site/` publish directory, adds a
   `.nojekyll` marker, and deploys it to GitHub Pages.
2. `app/static/index.html` becomes the **root `index.html`** of the published
   site, which is what fixes the previous `404 — File not found`.
3. It can also be run manually from the repo's **Actions** tab
   (`Deploy static site to GitHub Pages` → **Run workflow**).

**Published site root:** `index.html` (lowercase — matches the URL GitHub Pages
serves at `/`). No casing or path mismatch.

**Source of truth:** the page lives only in `app/static/`. There is no second
copy to keep in sync, so there is no ongoing manual maintenance.

## One-time GitHub setup

The workflow self-enables Pages (`actions/configure-pages` with
`enablement: true`), so in most cases no manual step is needed. If the first run
fails to enable it, set it once:

- **Settings → Pages → Build and deployment → Source = `GitHub Actions`**

Do **not** set the source to "Deploy from a branch" — that mode would look for an
`index.html` at the branch root (which is why the 404 happened) and would ignore
this workflow.

After a successful run, the site URL is shown in the workflow's `deploy` job and
under **Settings → Pages** (typically
`https://rippu-honwan.github.io/f125-demo/`).

## Verifying the deployment

1. Push to `main` (or trigger the workflow manually).
2. Open the **Actions** tab and confirm both `build` and `deploy` jobs are green.
3. Open the Pages URL — the F1 AI Driver Coach UI should load (no 404).

## Important: static site vs. backend

GitHub Pages serves **static files only**. The published page renders the full
UI, but its interactive analysis calls a Python/FastAPI backend that does **not**
run on GitHub Pages:

- `POST /laps` and `POST /analyze` (in `app/main.py`) require a running server.
- On the Pages URL these requests will fail; the page itself still loads.

To use the full analysis pipeline, run the backend locally (unchanged):

```bash
pip install -e ".[web]"
python -m app.main            # or: uvicorn app.main:app --reload
```

## Custom domain (CNAME)

A `CNAME` file was previously added and then removed from this repo. If you want
a custom domain again, add the domain under **Settings → Pages → Custom domain**
(or add a `CNAME` file inside `app/static/` so it is included in `_site/`).
The site works without one — it will be served from the default
`*.github.io` URL.
