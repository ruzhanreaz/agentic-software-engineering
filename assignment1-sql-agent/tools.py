import re
import sqlite3
import ollama

DB_PATH = "data/store.db"

DB_SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    city TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
"""


def _clean_sql(raw_sql: str) -> str:
    """Strip markdown code fences if the LLM includes them."""
    cleaned = re.sub(r"^```(?:sql)?\s*", "", raw_sql.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def generate_sql(
    question: str, db_path: str = DB_PATH, db_schema: str = DB_SCHEMA
) -> str:
    """Generate a SQLite query from a natural language question using Ollama."""
    prompt = (
        "You are an expert SQLite developer. Convert the user's question into a valid, "
        "read-only SQLite SELECT query based strictly on the schema below.\n"
        "Return ONLY the raw SQL query with no explanations or markdown formatting.\n\n"
        f"Schema:\n{db_schema}\n\n"
        f"Question: {question}\n"
        "SQL:"
    )

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
        )
        return _clean_sql(response.message.content)
    except Exception as e:
        return f"Error generating SQL: {str(e)}"


generate_sql_tool_schema = {
    "type": "function",
    "function": {
        "name": "generate_sql",
        "description": "Generate a SQLite query based on the user's natural language question.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question about the database.",
                }
            },
            "required": ["question"],
        },
    },
}


def execute_sql(query: str, db_path: str = DB_PATH, row_limit: int = 50) -> dict:
    """Execute a SQLite query safely and return its results or an error."""
    cleaned_query = _clean_sql(query)

    # Basic safeguard to prevent accidental data modification
    if not cleaned_query.upper().lstrip().startswith(("SELECT", "WITH", "PRAGMA")):
        return {
            "success": False,
            "error": "Security restriction: Only read-only queries (SELECT, WITH) are allowed.",
        }

    try:
        with sqlite3.connect(db_path) as connection:
            cursor = connection.execute(cleaned_query)

            if cursor.description is None:
                return {
                    "success": True,
                    "row_count": cursor.rowcount,
                }

            columns = [column[0] for column in cursor.description]
            rows = [
                dict(zip(columns, row))
                for row in cursor.fetchmany(max(0, row_limit))
            ]
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }
    except sqlite3.Error as error:
        return {"success": False, "error": str(error)}


execute_sql_tool_schema = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Execute a SQLite query on the store database and return its results or an error.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The valid SQLite query to execute.",
                },
                "row_limit": {
                    "type": "integer",
                    "description": "The maximum number of rows to return (default 50).",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_SCHEMAS = [generate_sql_tool_schema, execute_sql_tool_schema]
TOOL_REGISTRY = {
    "generate_sql": generate_sql,
    "execute_sql": execute_sql,
}