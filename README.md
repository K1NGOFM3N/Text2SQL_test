# Ask your database

A text-to-SQL web app powered by Claude. People enter credentials for their own
PostgreSQL, MySQL/MariaDB, SQL Server (or SQLite) database, ask questions in plain
language, and get back the SQL and the results.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your Anthropic API key in .env
uvicorn app:app --reload
```

Open http://localhost:8000.

## How it works

1. **Connect.** The browser sends the credentials to the server once. The server opens a
   connection pool, reads the schema (tables, columns, primary and foreign keys) and returns
   a random session token. The password field is cleared in the browser.
2. **Ask.** The question, the schema and up to six earlier question/SQL pairs (so follow-ups
   like "now break that down by month" work) go to Claude, which answers through a forced
   tool call so the SQL always comes back in a structured form.
3. **Run.** The SQL is checked and run read-only. If it fails, the error goes back to Claude
   once for a corrected query.
4. **Edit.** The SQL is editable; "Run SQL" re-runs your version through the same checks.

## Security

Four layers keep queries read-only, from most to least important:

- **Use a read-only database user.** This is the only guarantee. Everything else is a backstop.
- Each query runs in a transaction that is always rolled back, set to `READ ONLY` on
  PostgreSQL and MySQL, and with `PRAGMA query_only` on SQLite.
- SQL is parsed with sqlglot: one statement only, must be a query, and no INSERT, UPDATE,
  DELETE, DDL, SELECT INTO, row locks or risky functions (file access, sleep, dblink, xp_cmdshell…).
- Timeouts per query and a row cap (`MAX_ROWS`).

Other things to know before hosting this for other people:

- **Credentials** live only in server memory for the session (default 30 minutes idle) and
  are never logged or written to disk. Serve the app over HTTPS so they're encrypted in transit.
- **Network reach.** The server connects to whatever host a user types, which means users can
  probe your internal network from it. Set `ALLOWED_DB_HOSTS` in production.
- **What Claude sees.** The schema, the questions and any database error messages are sent to
  the Anthropic API. Query *results* are not.
- **Sessions** are held in memory, so run a single worker process (the default for
  `uvicorn app:app`), or move sessions to a shared store if you scale out.
- There is no user login. Anyone who can reach the page can connect to databases they hold
  credentials for. Put it behind your own auth if it's public.

## Adding another database

Add an entry to `DIALECTS` in `app.py` with its SQLAlchemy driver, default port and sqlglot
dialect name (e.g. Snowflake, BigQuery, Oracle, DuckDB), and add its driver to `requirements.txt`.
