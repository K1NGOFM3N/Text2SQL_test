"""
Ask your database: text-to-SQL with Claude, as a Streamlit app.

Local:      streamlit run streamlit_app.py
Deploy:     Streamlit Community Cloud, main file = streamlit_app.py,
            ANTHROPIC_API_KEY in the app's Secrets.
"""

import math
import os
import tempfile

import anthropic
import certifi
import pandas as pd
import sqlglot
import streamlit as st
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
from sqlglot import exp

st.set_page_config(page_title="Ask your database", page_icon="🗄️", layout="wide")


# ---------------------------------------------------------------- settings ---
def setting(name, default=None):
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file at all
        pass
    return os.getenv(name, default)


MODEL = setting("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_ROWS = int(setting("MAX_ROWS", 500))
QUERY_TIMEOUT_S = int(setting("QUERY_TIMEOUT_SECONDS", 30))
MAX_SCHEMA_TABLES = int(setting("MAX_SCHEMA_TABLES", 150))
ALLOWED_HOSTS = {h.strip().lower() for h in str(setting("ALLOWED_DB_HOSTS", "")).split(",") if h.strip()}
HISTORY_TURNS = 6

DIALECTS = {
    "postgresql": {"driver": "postgresql+psycopg", "port": 5432, "sqlglot": "postgres", "label": "PostgreSQL"},
    "mysql": {"driver": "mysql+pymysql", "port": 3306, "sqlglot": "mysql", "label": "MySQL / MariaDB"},
    "mssql": {"driver": "mssql+pymssql", "port": 1433, "sqlglot": "tsql", "label": "SQL Server"},
    "sqlite": {"driver": "sqlite", "port": None, "sqlglot": "sqlite", "label": "SQLite (upload a file)"},
}

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("history", [])


def clean_error(e: Exception) -> str:
    msg = str(getattr(e, "orig", None) or e)
    return msg.split("\n[SQL:")[0].strip()[:800]


# ----------------------------------------------------------------- connect ---
def build_engine(dialect, host, port, database, username, password, ssl, sqlite_path=None):
    d = DIALECTS[dialect]
    if dialect == "sqlite":
        return create_engine(URL.create("sqlite", database=sqlite_path))
    if not host:
        raise ValueError("Enter a host.")
    if ALLOWED_HOSTS and host.lower() not in ALLOWED_HOSTS:
        raise ValueError(f"Host {host} isn't on this app's allowed list.")
    url = URL.create(d["driver"], username=username or None, password=password or None,
                     host=host, port=port or d["port"], database=database)
    args = {}
    if dialect == "postgresql":
        args = {"connect_timeout": 10, "options": f"-c statement_timeout={QUERY_TIMEOUT_S * 1000}"}
        if ssl:
            args["sslmode"] = "require"
    elif dialect == "mysql":
        args = {"connect_timeout": 10, "read_timeout": QUERY_TIMEOUT_S}
        if ssl:
            args["ssl"] = {"ca": certifi.where()}
    elif dialect == "mssql":
        args = {"login_timeout": 10, "timeout": QUERY_TIMEOUT_S}
    return create_engine(url, connect_args=args, pool_pre_ping=True, pool_size=2, max_overflow=2)


def describe_schema(engine, schema=None):
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
            bit = f"{c['name']} {ctype}" + (" PK" if c["name"] in pk else "")
            if c["name"] in fk_map:
                bit += f" -> {fk_map[c['name']]}"
            parts.append(bit)
            tree_cols.append((c["name"], ctype, c["name"] in pk))
        lines.append(f"{kind.upper()} {qualified}({', '.join(parts)})")
        tree.append({"name": qualified, "kind": kind, "columns": tree_cols})
    if len(names) > MAX_SCHEMA_TABLES:
        lines.append(f"-- {len(names) - MAX_SCHEMA_TABLES} more tables omitted")
    return "\n".join(lines), tree


def disconnect():
    if "engine" in ss:
        ss.engine.dispose()
    if ss.get("tmp_path"):
        try:
            os.remove(ss.tmp_path)
        except OSError:
            pass
    for k in ["engine", "dialect", "schema_text", "tables", "label", "tmp_path"]:
        ss.pop(k, None)
    ss.messages, ss.history = [], []


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


def validate_readonly(sql, dialect):
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


def run_query(sql):
    """Returns (DataFrame, truncated). Raises on failure."""
    validate_readonly(sql, ss.dialect)
    raw = {"no_parameters": True}
    with ss.engine.connect() as conn:
        trans = conn.begin()
        try:
            if ss.dialect in ("postgresql", "mysql"):
                conn.exec_driver_sql("SET TRANSACTION READ ONLY", execution_options=raw)
            elif ss.dialect == "sqlite":
                conn.exec_driver_sql("PRAGMA query_only = ON", execution_options=raw)
            result = conn.exec_driver_sql(sql, execution_options=raw)
            if not result.returns_rows:
                return pd.DataFrame(), False
            columns = list(result.keys())
            rows = result.fetchmany(MAX_ROWS + 1)
        finally:
            trans.rollback()
    # Streamlit can't display duplicate column names, so number repeats.
    seen, unique = {}, []
    for c in columns:
        seen[c] = seen.get(c, 0) + 1
        unique.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df = pd.DataFrame([[to_cell(v) for v in r] for r in rows[:MAX_ROWS]], columns=unique)
    return df, len(rows) > MAX_ROWS


# ------------------------------------------------------------------ Claude ---
SQL_TOOL = {
    "name": "submit_sql",
    "description": "Submit the SQL query that answers the user's question, or explain why it can't be answered.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "One read-only SELECT query in the target dialect. "
                                                     "Empty string if the schema can't answer the question."},
            "explanation": {"type": "string", "description": "One or two plain-language sentences: what the "
                                                             "query returns, or why it can't be answered."},
        },
        "required": ["sql", "explanation"],
    },
}


def api_key():
    return setting("ANTHROPIC_API_KEY") or ss.get("user_api_key")


def ask_claude(question, failed=None):
    label = DIALECTS[ss.dialect]["label"].split(" (")[0]
    system = f"""You translate questions about a {label} database into SQL.

Database schema (TABLE/VIEW name(column type [PK] [-> referenced table.column])):
{ss.schema_text}

Rules:
- Write exactly one read-only query (SELECT, optionally with CTEs). Never modify data or schema.
- Use only tables and columns that appear in the schema above. Use {label} syntax and functions.
- Unless the question asks for an aggregate or a specific count, limit results to {MAX_ROWS} rows.
- Prefer clear column aliases and sensible ORDER BY. Qualify columns when joining.
- When matching text values you haven't seen, use case-insensitive matching.
- If the question is ambiguous, pick the most reasonable reading and say which in the explanation.
- If the schema can't answer the question, return an empty sql and explain what's missing.
- Always respond by calling submit_sql."""

    messages = []
    for q, sql in ss.history[-HISTORY_TURNS:]:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": f"```sql\n{sql}\n```"}]
    content = question
    if failed:
        content += (f"\n\nYour previous attempt:\n```sql\n{failed[0]}\n```\nfailed with this error:\n"
                    f"{failed[1]}\n\nWrite a corrected query.")
    messages.append({"role": "user", "content": content})

    client = anthropic.Anthropic(api_key=api_key())
    resp = client.messages.create(
        model=MODEL, max_tokens=2000, system=system, messages=messages,
        tools=[SQL_TOOL], tool_choice={"type": "tool", "name": "submit_sql"},
    )
    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if not block:
        raise RuntimeError("Claude didn't return a query. Try rephrasing the question.")
    return (block.input.get("sql") or "").strip().rstrip(";"), block.input.get("explanation", "")


def answer(question):
    """Generate SQL, run it, retry once on failure. Returns a message dict."""
    msg = {"role": "assistant", "sql": "", "explanation": "", "df": None,
           "truncated": False, "error": None, "repaired": False}
    try:
        msg["sql"], msg["explanation"] = ask_claude(question)
    except anthropic.AuthenticationError:
        msg["error"] = "The Anthropic API key was rejected. Check it and try again."
        return msg
    except Exception as e:
        msg["error"] = f"Couldn't reach Claude: {e}"
        return msg
    if not msg["sql"]:
        return msg

    try:
        msg["df"], msg["truncated"] = run_query(msg["sql"])
    except Exception as first:
        err = str(first) if isinstance(first, ValueError) else clean_error(first)
        try:
            msg["sql"], msg["explanation"] = ask_claude(question, failed=(msg["sql"], err))
            msg["repaired"] = True
            if msg["sql"]:
                msg["df"], msg["truncated"] = run_query(msg["sql"])
        except Exception as second:
            msg["error"] = str(second) if isinstance(second, ValueError) else clean_error(second)
            return msg

    ss.history.append((question, msg["sql"]))
    return msg


# ---------------------------------------------------------------------- UI ---
def render(m):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
            return
        if m.get("explanation"):
            st.markdown(m["explanation"])
        if m.get("sql"):
            st.code(m["sql"], language="sql")
        if m.get("repaired"):
            st.caption("Fixed after a first attempt failed.")
        if m.get("error"):
            st.error(m["error"])
        df = m.get("df")
        if df is not None:
            if df.shape[1] == 0:
                st.caption("The query ran but returned no columns.")
            else:
                n = len(df)
                note = f" (showing the first {n:,}; add filters or a LIMIT to see others)" if m["truncated"] else ""
                st.caption(f"{n:,} row{'s' if n != 1 else ''}{note}")
                st.dataframe(df, hide_index=True)


with st.sidebar:
    st.header("Ask your database")

    if not setting("ANTHROPIC_API_KEY"):
        ss.user_api_key = st.text_input(
            "Anthropic API key", type="password", value=ss.get("user_api_key", ""),
            help="Used only for this session. The app owner can set ANTHROPIC_API_KEY in Secrets instead.",
        )

    if "engine" not in ss:
        dialect = st.selectbox("Database type", list(DIALECTS), format_func=lambda k: DIALECTS[k]["label"])
        with st.form("connect"):
            if dialect == "sqlite":
                upload = st.file_uploader("SQLite database file", type=["db", "sqlite", "sqlite3"])
                host = port = username = password = schema = None
                database, ssl = "", False
            else:
                upload = None
                c1, c2 = st.columns([3, 1])
                host = c1.text_input("Host", placeholder="db.example.com")
                port = c2.number_input("Port", value=DIALECTS[dialect]["port"], step=1)
                database = st.text_input("Database name")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                schema = st.text_input("Schema (optional)",
                                       placeholder={"postgresql": "public", "mssql": "dbo"}.get(dialect, ""))
                ssl = st.checkbox("Require SSL")
            submitted = st.form_submit_button("Connect", type="primary", use_container_width=True)

        st.caption("Use a read-only database user. Credentials are kept only for this browser "
                   "session and never saved. The database must be reachable from the internet.")

        if submitted:
            tmp_path = None
            try:
                if dialect == "sqlite":
                    if not upload:
                        raise ValueError("Choose a SQLite file to upload.")
                    data = upload.getvalue()
                    if not data.startswith(b"SQLite format 3\x00"):
                        raise ValueError("That file isn't a SQLite database.")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
                        f.write(data)
                        tmp_path = f.name
                    database = upload.name
                elif not database:
                    raise ValueError("Enter a database name.")
                with st.spinner("Connecting and reading the schema"):
                    engine = build_engine(dialect, host, int(port or 0) or None, database,
                                          username, password, ssl, sqlite_path=tmp_path)
                    try:
                        with engine.connect() as conn:
                            conn.exec_driver_sql("SELECT 1", execution_options={"no_parameters": True})
                        schema_text, tree = describe_schema(engine, schema or None)
                    except Exception:
                        engine.dispose()
                        raise
                if not tree:
                    engine.dispose()
                    raise ValueError("Connected, but found no tables you can read. Check the database or schema name.")
                ss.engine, ss.dialect, ss.schema_text, ss.tables = engine, dialect, schema_text, tree
                ss.label = database if dialect == "sqlite" else f"{database} on {host}"
                ss.tmp_path = tmp_path
                st.rerun()
            except Exception as e:
                if tmp_path:
                    os.remove(tmp_path)
                st.error(str(e) if isinstance(e, ValueError) else f"Couldn't connect: {clean_error(e)}")
    else:
        st.success(f"Connected to **{ss.label}**")
        st.caption(f"{DIALECTS[ss.dialect]['label'].split(' (')[0]}, {len(ss.tables)} tables and views")
        if st.button("Disconnect", use_container_width=True):
            disconnect()
            st.rerun()

        st.subheader("Tables")
        flt = st.text_input("Filter tables", label_visibility="collapsed", placeholder="Filter tables").lower()
        for t in ss.tables:
            if flt and flt not in t["name"].lower() and not any(flt in c[0].lower() for c in t["columns"]):
                continue
            with st.expander(t["name"] + (" (view)" if t["kind"] == "view" else "")):
                st.markdown("\n".join(f"- `{n}` {typ}{' **PK**' if pk else ''}" for n, typ, pk in t["columns"]))


connected = "engine" in ss
if not connected:
    st.info("Connect to a database in the sidebar to start asking questions.")
elif not api_key():
    st.warning("Enter an Anthropic API key in the sidebar.")
elif not ss.messages:
    st.markdown("Ask a question about your data. Follow-up questions remember the earlier ones.")

for m in ss.messages:
    render(m)

question = st.chat_input("Which customers spent the most last quarter?",
                         disabled=not (connected and api_key()))
if question:
    user_msg = {"role": "user", "content": question}
    ss.messages.append(user_msg)
    render(user_msg)
    with st.spinner("Writing and running SQL"):
        reply = answer(question)
    ss.messages.append(reply)
    render(reply)

# Edit and re-run the most recent query.
last_sql = next((m["sql"] for m in reversed(ss.messages) if m.get("sql")), "")
if connected and last_sql:
    with st.expander("Edit and re-run the last query"):
        edited = st.text_area("SQL", value=last_sql, height=160, key=f"edit_{len(ss.messages)}",
                              label_visibility="collapsed")
        if st.button("Run SQL"):
            msg = {"role": "assistant", "sql": edited.strip().rstrip(";"),
                   "explanation": "Ran your edited query.", "df": None, "truncated": False,
                   "error": None, "repaired": False}
            try:
                msg["df"], msg["truncated"] = run_query(msg["sql"])
            except Exception as e:
                msg["error"] = str(e) if isinstance(e, ValueError) else clean_error(e)
            ss.messages.append(msg)
            st.rerun()
