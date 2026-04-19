# Task: WordPress Blog App

Build a Flask web app with SQLite storage that supports CRUD operations for blog posts via an admin interface and displays published posts on a public-facing home page.

## What to build

Create a Python application that:

1. **Entry point**: Single `app.py` file, startable with `uv run python app.py`.
2. **Database**: SQLite database file named `blog.db` in the workspace directory. Schema must include at least a `posts` table with columns: `id` (integer primary key), `title` (text, required), `content` (text, required), `created_at` (timestamp), and `is_published` (boolean, default False).
3. **Admin interface** (`/admin`): A page where users can:
   - View all posts (published and draft)
   - Create a new post (title + content)
   - Edit an existing post
   - Delete a post
   - Mark posts as published or draft
4. **Public interface** (`/`): A home page displaying all published posts with title and content. Draft posts must not be visible.
5. **Port**: Listen on port 8181.

## Pass condition

The grader will:

1. Start your app with `uv run python app.py`
2. Verify HTTP 200 on `/` and `/admin`
3. Manually verify:
   - Admin page allows creating a post
   - Created posts appear in admin interface
   - Only published posts appear on the public home page
   - Posts can be edited and deleted from the admin interface
   - Refreshing the page persists changes (data is saved to the database)

## Guidance

- Use Flask for routing and HTML templates (Jinja2).
- Use Python's `sqlite3` module directly for database access (no ORM required, but SQLAlchemy is fine if you prefer).
- Keep the UI simple and functional — no CSS framework required, but basic styling is welcome.
- A minimal working solution is ~150–200 lines of Python + a few Jinja2 templates.
- Before declaring done, manually test: create a post, mark it published, refresh the public page and verify it appears; mark it draft and verify it disappears.
- Dependency bootstrap: `uv` will initialize the project on first run; install Flask with your preferred method (add to `pyproject.toml` or use `uv pip install flask`).

## Hard rules

The entry point must be `app.py`, startable with `uv run python app.py`. Do not leave a stock `main.py` alongside it.

The server must listen on port 8181.

The database must be SQLite, stored as `blog.db` in the workspace root (not in a subdirectory).

No external APIs or network calls. All data is local.

Provide a working admin interface and public interface. Both must be reachable and functional by the time the task is declared complete.

No authentication required for this task — the admin and public interfaces can both be openly accessible.

Write code and run shell commands. Do not write long plans, design documents, or list-item labels into your chat replies — anything that needs to land on disk should be written into a file.
