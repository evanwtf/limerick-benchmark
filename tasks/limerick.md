# Task: Limerick Web App

Build a Flask web app that serves a random limerick and auto-refreshes
it every 5 seconds.

## What to build

- A Flask app in **`app.py`** that listens on **port 8181**.
- The root URL `/` must return HTTP 200 with an HTML page containing one
  limerick and a mechanism that replaces it with a different one every
  5 seconds.
- The 5-second refresh may use JavaScript `setInterval` (fetching a new
  limerick from a second endpoint) or a `<meta http-equiv="refresh"
  content="5">` tag. Either is acceptable.

## Pass condition

Your task is complete when:

```
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8181
```

prints `200`, and the returned HTML contains a 5-line limerick plus
either a `setInterval` call or a `meta refresh` tag.

The grader does NOT judge the quality, originality, or rhyme of the
limericks. It only checks that the server responds.

## Limerick source

A file named **`limericks.txt`** is already present in the workspace.
It contains 20 limericks separated by blank lines. Your app should
read that file on startup and pick one at random on each request.

You do NOT need to invent new limericks. Reading from `limericks.txt`
is the expected approach.

## Recommended order of work

1. Write a minimal `app.py` that reads `limericks.txt`, picks one at
   random, and returns it wrapped in an HTML page with a meta refresh
   tag. Keep it small — under 50 lines.
2. Start it with `uv run python app.py &` and verify with
   `curl -sS http://localhost:8181`. Confirm you see `200` and a
   limerick.
3. Stop, and you are done.

## Hard rules

- **Write code, not limericks.** Do not output limerick text in your
  chat responses. Put it in code and files. The grader never reads
  your chat.
- **The entry point is `app.py`** — startable with
  `uv run python app.py`. Do not leave a stock `main.py` alongside it.
  If `main.py` exists from `uv init`, delete it or overwrite it so only
  `app.py` is the runnable entry point.
- **Port 8181**, not 5000 or 8000.
- **No external APIs, no network calls** at runtime — read from the
  local `limericks.txt`.
- **Verify before declaring done.** Actually curl the server and see a
  200 before you stop.

## Limerick format (for reference)

A limerick is a 5-line AABBA poem. The ones in `limericks.txt` already
follow this format; you do not need to validate or re-rhyme them.
