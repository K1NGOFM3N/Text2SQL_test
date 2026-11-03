"""
Text-to-SQL with Claude.

Users connect to a database of their choice, ask questions in plain language,
Claude writes the SQL, and the app runs it read-only and shows the results.

Run:  uvicorn app:app --reload
"""

import math
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import anthropic
import certifi
import sqlglot
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL, Engine
from sqlglot import exp

load_dotenv()

# ---------------------------------------------------------------- settings ---
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_ROWS = int(os.getenv("MAX_ROWS", "500"))
QUERY_TIMEOUT_S = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))
SESSION_TTL_S = int(os.getenv("SESSION_TTL_MINUTES", "30")) * 60
ALLOW_SQLITE = os.getenv("ALLOW_SQLITE", "false").lower() == "true"
ALLOWED_HOSTS = {h.strip().lower() for h in os.getenv("ALLOWED_DB_HOSTS", "").split(",") if h.strip()}
MAX_SCHEMA_TABLES = int(os.getenv("MAX_SCHEMA_TABLES", "150"))
HISTORY_TURNS = 6  # previous question/SQL pairs sent to Claude for follow-ups

DIALECTS = {
    "postgresql": {"driver": "postgresql+psycopg", "port": 5432, "sqlglot": "postgres", "label": "PostgreSQL"},
    "mysql": {"driver": "mysql+pymysql", "port": 3306, "sqlglot": "mysql", "label": "MySQL / MariaDB"},
    "mssql": {"driver": "mssql+pymssql", "port": 1433, "sqlglot": "tsql", "label": "SQL Server"},
    "sqlite": {"driver": "sqlite", "port": None, "sqlglot": "sqlite", "label": "SQLite"},
}

claude = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
app = FastAPI(title="Text-to-SQL")


# ---------------------------------------------------------------- sessions ---
class Session:
    def __init__(self, engine: Engine, dialect: str, schema_text: str, label: str):
        self.engine = engine
        self.dialect = dialect
        self.schema_text = schema_text
        self.label = label
        self.history: list[tuple[str, str]] = []
        self.last_used = time.time()


_sessions: dict[str, Session] = {}
_lock = threading.Lock()


def _cleanup_expired() -> None:
    now = time.time()
    with _lock:
        expired = [k for k, s in _sessions.items() if now - s.last_used > SESSION_TTL_S]
        for k in expired:
            _sessions.pop(k).engine.dispose()


def get_session(token: Optional[str]) -> Session:
    _cleanup_expired()
    with _lock:
        s = _sessions.get(token or "")
    if not s:
        raise HTTPException(401, "Not connected, or the session expired. Connect to a database again.")
    s.last_used = time.time()
    return s


def clean_error(e: Exception) -> str:
    """Driver error text without SQLAlchemy's boilerplate (never includes the password)."""
    msg = str(getattr(e, "orig", None) or e)
    return msg.split("\n[SQL:")[0].strip()[:800]


# ----------------------------------------------------------------- connect ---
class ConnectRequest(BaseModel):
    dialect: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    schema_name: Optional[str] = None
    ssl: bool = False


def build_engine(req: ConnectRequest) -> Engine:
    if req.dialect not in DIALECTS:
        raise HTTPException(400, f"Unsupported database type: {req.dialect}")
    d = DIALECTS[req.dialect]

    if req.dialect == "sqlite":
        if not ALLOW_SQLITE:
            raise HTTPException(400, "SQLite files are turned off on this server (set ALLOW_SQLITE=true).")
        return create_engine(URL.create("sqlite", database=req.database))

    if not req.host:
        raise HTTPException(400, "Enter a host.")
    if ALLOWED_HOSTS and req.host.lower() not in ALLOWED_HOSTS:
        raise HTTPException(400, f"Host {req.host} isn't on this server's allowed list.")

    url = URL.create(
        d["driver"], username=req.username or None, password=req.password or None,
        host=req.host, port=req.port or d["port"], database=req.database,
    )
    args: dict = {}
    if req.dialect == "postgresql":
        args = {"connect_timeout": 10, "options": f"-c statement_timeout={QUERY_TIMEOUT_S * 1000}"}
        if req.ssl:
            args["sslmode"] = "require"
    elif req.dialect == "mysql":
        args = {"connect_timeout": 10, "read_timeout": QUERY_TIMEOUT_S}
        if req.ssl:
            args["ssl"] = {"ca": certifi.where()}
    elif req.dialect == "mssql":
        args = {"login_timeout": 10, "timeout": QUERY_TIMEOUT_S}
    return create_engine(url, connect_args=args, pool_pre_ping=True, pool_size=2, max_overflow=2)


def describe_schema(engine: Engine, schema: Optional[str]) -> tuple[str, list[dict]]:
    """Compact text description for Claude plus a tree for the sidebar."""
    insp = inspect(engine)
    names = [(t, "table") for t in insp.get_table_names(schema=schema)]
    try:
        names += [(v, "view") for v in insp.get_view_names(schema=schema)]
    except Exception:
        pass

    lines, tree = [], []
    for name, kind in names[:MAX_SCHEMA_TABLES]:
        qualified = f"{schema}.{name}" if schema else name
        try:
            cols = insp.get_columns(name, schema=schema)
            pk = set(insp.get_pk_constraint(name, schema=schema).get("constrained_columns") or [])
            fks = insp.get_foreign_keys(name, schema=schema) if kind == "table" else []
        except Exception:
            continue
        fk_map = {}
        for fk in fks:
            ref = fk.get("referred_table")
            if fk.get("referred_schema"):
                ref = f"{fk['referred_schema']}.{ref}"
            for c, rc in zip(fk.get("constrained_columns", []), fk.get("referred_columns", [])):
                fk_map[c] = f"{ref}.{rc}"

        parts, tree_cols = [], []
        for c in cols:
            ctype = str(c["type"]) if c.get("type") is not None else "?"
            bit = f"{c['name']} {ctype}"
            if c["name"] in pk:
                bit += " PK"
            if c["name"] in fk_map:
                bit += f" -> {fk_map[c['name']]}"
            parts.append(bit)
            tree_cols.append({"name": c["name"], "type": ctype, "pk": c["name"] in pk})
        lines.append(f"{kind.upper()} {qualified}({', '.join(parts)})")
        tree.append({"name": qualified, "kind": kind, "columns": tree_cols})

    if len(names) > MAX_SCHEMA_TABLES:
        lines.append(f"-- {len(names) - MAX_SCHEMA_TABLES} more tables omitted")
    return "\n".join(lines), tree


@app.post("/api/connect")
def connect(req: ConnectRequest):
    engine = build_engine(req)
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1", execution_options={"no_parameters": True})
        schema_text, tree = describe_schema(engine, req.schema_name or None)
    except HTTPException:
        raise
    except Exception as e:
        engine.dispose()
        raise HTTPException(400, f"Couldn't connect: {clean_error(e)}")

    if not tree:
        engine.dispose()
        raise HTTPException(400, "Connected, but found no tables you can read. Check the database or schema name.")

    label = req.database if req.dialect == "sqlite" else f"{req.database} on {req.host}"
    token = secrets.token_urlsafe(32)
    with _lock:
        _sessions[token] = Session(engine, req.dialect, schema_text, label)
    return {"token": token, "label": label, "dialect": DIALECTS[req.dialect]["label"], "tables": tree}


@app.post("/api/disconnect")
def disconnect(x_session_token: Optional[str] = Header(None)):
    with _lock:
        s = _sessions.pop(x_session_token or "", None)
    if s:
        s.engine.dispose()
    return {"ok": True}


# -------------------------------------------------------------- safety net ---
FORBIDDEN_NODES = tuple(
    getattr(exp, n) for n in
    ["Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Merge",
     "Command", "Into", "Lock", "TruncateTable", "Grant", "Set", "Use", "Copy"]
    if hasattr(exp, n)
)
FORBIDDEN_FUNCS = {
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "lo_import", "lo_export",
    "dblink", "dblink_exec", "pg_terminate_backend", "pg_cancel_backend", "set_config", "pg_sleep",
    "load_file", "sleep", "benchmark", "xp_cmdshell", "openrowset", "opendatasource", "openquery",
}
QUERY_TYPES = (exp.Query,) if hasattr(exp, "Query") else (exp.Select, exp.Union, exp.Intersect, exp.Except)


def validate_readonly(sql: str, dialect: str) -> None:
    """Reject anything that isn't a single read-only query. The DB user should be read-only too."""
    try:
        stmts = [s for s in sqlglot.parse(sql, read=DIALECTS[dialect]["sqlglot"]) if s is not None]
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"Couldn't parse the SQL: {str(e).splitlines()[0]}")
    if len(stmts) != 1:
        raise ValueError("Only one statement can run at a time.")
    stmt = stmts[0]
    if not isinstance(stmt, QUERY_TYPES):
        raise ValueError("Only read-only queries (SELECT) can run here.")
    for node in stmt.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"Only read-only queries can run here ({type(node).__name__} isn't allowed).")
        if isinstance(node, exp.Func):
            fname = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if fname in FORBIDDEN_FUNCS:
                raise ValueError(f"The function {fname}() isn't allowed.")


def to_cell(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(v))} bytes>"
    return str(v)


def run_query(s: Session, sql: str) -> dict:
    validate_readonly(sql, s.dialect)
    raw = {"no_parameters": True}
    with s.engine.connect() as conn:
        trans = conn.begin()
        try:
            if s.dialect == "postgresql":
                conn.exec_driver_sql("SET TRANSACTION READ ONLY", execution_options=raw)
            elif s.dialect == "mysql":
                conn.exec_driver_sql("SET TRANSACTION READ ONLY", execution_options=raw)
            elif s.dialect == "sqlite":
                conn.exec_driver_sql("PRAGMA query_only = ON", execution_options=raw)
            result = conn.exec_driver_sql(sql, execution_options=raw)
            if not result.returns_rows:
                return {"columns": [], "rows": [], "truncated": False}
            columns = list(result.keys())
            rows = result.fetchmany(MAX_ROWS + 1)
        finally:
            trans.rollback()
    return {
        "columns": columns,
        "rows": [[to_cell(v) for v in r] for r in rows[:MAX_ROWS]],
        "truncated": len(rows) > MAX_ROWS,
    }


# ------------------------------------------------------------------ Claude ---
SQL_TOOL = {
    "name": "submit_sql",
    "description": "Submit the SQL query that answers the user's question, or explain why it can't be answered.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "One read-only SELECT query in the target dialect. Empty string if the "
                               "question can't be answered from this schema.",
            },
            "explanation": {
                "type": "string",
                "description": "One or two plain-language sentences: what the query returns, or why "
                               "the question can't be answered.",
            },
        },
        "required": ["sql", "explanation"],
    },
}


def system_prompt(s: Session) -> str:
    label = next(d["label"] for k, d in DIALECTS.items() if k == s.dialect)
    return f"""You translate questions about a {label} database into SQL.

Database schema (TABLE/VIEW name(column type [PK] [-> referenced table.column])):
{s.schema_text}

Rules:
- Write exactly one read-only query (SELECT, optionally with CTEs). Never modify data or schema.
- Use only tables and columns that appear in the schema above. Use {label} syntax and functions.
- Unless the question asks for an aggregate or a specific count, limit results to {MAX_ROWS} rows.
- Prefer clear column aliases and sensible ORDER BY. Qualify columns when joining.
- When matching text values you haven't seen, use case-insensitive matching.
- If the question is ambiguous, pick the most reasonable reading and say which in the explanation.
- If the schema can't answer the question, return an empty sql and explain what's missing.
- Always respond by calling submit_sql."""


def ask_claude(s: Session, question: str, failed: Optional[tuple[str, str]] = None) -> dict:
    messages = []
    for q, sql in s.history[-HISTORY_TURNS:]:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": f"```sql\n{sql}\n```"}]
    content = question
    if failed:
        content += (f"\n\nYour previous attempt:\n```sql\n{failed[0]}\n```\nfailed with this error:\n"
                    f"{failed[1]}\n\nWrite a corrected query.")
    messages.append({"role": "user", "content": content})

    resp = claude.messages.create(
        model=MODEL, max_tokens=2000, system=system_prompt(s), messages=messages,
        tools=[SQL_TOOL], tool_choice={"type": "tool", "name": "submit_sql"},
    )
    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if not block:
        raise HTTPException(502, "Claude didn't return a query. Try rephrasing the question.")
    return {"sql": (block.input.get("sql") or "").strip().rstrip(";"),
            "explanation": block.input.get("explanation", "")}


# -------------------------------------------------------------- endpoints ---
class AskRequest(BaseModel):
    question: str


class RunRequest(BaseModel):
    sql: str


@app.post("/api/ask")
def ask(req: AskRequest, x_session_token: Optional[str] = Header(None)):
    s = get_session(x_session_token)
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Type a question first.")
    try:
        gen = ask_claude(s, question)
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {e.message if hasattr(e, 'message') else e}")

    out = {**gen, "columns": [], "rows": [], "truncated": False, "error": None, "repaired": False}
    if not gen["sql"]:
        return out

    try:
        out.update(run_query(s, gen["sql"]))
    except Exception as first_err:
        # One repair attempt: show Claude the error and let it fix the query.
        err = str(first_err) if isinstance(first_err, ValueError) else clean_error(first_err)
        try:
            fixed = ask_claude(s, question, failed=(gen["sql"], err))
            out.update(fixed, repaired=True)
            if fixed["sql"]:
                out.update(run_query(s, fixed["sql"]))
        except Exception as second_err:
            out["error"] = str(second_err) if isinstance(second_err, ValueError) else clean_error(second_err)
            return out

    s.history.append((question, out["sql"]))
    return out


@app.post("/api/run")
def run(req: RunRequest, x_session_token: Optional[str] = Header(None)):
    s = get_session(x_session_token)
    try:
        return {**run_query(s, req.sql.strip().rstrip(";")), "error": None}
    except ValueError as e:
        return {"columns": [], "rows": [], "truncated": False, "error": str(e)}
    except Exception as e:
        return {"columns": [], "rows": [], "truncated": False, "error": clean_error(e)}


@app.get("/api/config")
def config():
    dialects = [{"id": k, "label": v["label"], "port": v["port"]}
                for k, v in DIALECTS.items() if k != "sqlite" or ALLOW_SQLITE]
    return {"dialects": dialects, "model": MODEL, "max_rows": MAX_ROWS}


STATIC = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
