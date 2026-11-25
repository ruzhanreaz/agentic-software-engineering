from agent import Agent
from tools import TOOL_SCHEMAS, TOOL_REGISTRY

SYSTEM_PROMPT = """You are a helpful data analyst assistant connected to a SQLite store database.
When a user asks a question about customers, products, orders, or sales:
1. First, call the `generate_sql` tool with the user's question to create the appropriate SQL query.
2. Next, call the `execute_sql` tool using the generated SQL query to fetch the data from the database.
3. If `execute_sql` returns an error, inspect the error, fix the SQL query, and call `execute_sql` again.
4. Finally, summarize the retrieved data clearly and concisely in plain English for the user.

Do not guess database numbers—always query the database using your tools."""


def main():
    agent = Agent(
        model="qwen2.5:3b",
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        tool_registry=TOOL_REGISTRY,
    )

    print("Agent initialized with qwen2.5:3b. Type 'exit' or 'quit' to stop.\n")

    # Example single query test
    sample_query = "How many customers are there?"
    print(f"User: {sample_query}")
    response = agent(sample_query)
    print(f"Agent: {response}\n")

    # Interactive loop
    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        reply = agent(user_input)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()