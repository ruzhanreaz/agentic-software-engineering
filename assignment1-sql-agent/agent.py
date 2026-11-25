import json
import ollama


class Agent:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        system: str = "",
        tools: list = None,
        tool_registry: dict = None,
        max_iterations: int = 5,
    ) -> None:
        self.model = model
        self.system = system
        self.messages: list = []
        self.tools = tools if tools is not None else []
        self.tool_registry = tool_registry if tool_registry is not None else {}
        self.max_iterations = max_iterations

        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message: str = "") -> str:
        if message:
            self.messages.append({"role": "user", "content": message})

        final_assistant_content = self.execute()

        if final_assistant_content:
            self.messages.append(
                {"role": "assistant", "content": final_assistant_content}
            )

        return final_assistant_content

    def execute(self) -> str:
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=self.tools if self.tools else None,
            )

            response_message = response.message

            # Check if the model decided to invoke any tools
            if response_message.tool_calls:
                # Append the assistant's tool-call request to conversation history
                self.messages.append(response_message.model_dump())

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments

                    # Ensure arguments are a dictionary (in case returned as a JSON string)
                    if isinstance(function_args, str):
                        try:
                            function_args = json.loads(function_args)
                        except json.JSONDecodeError:
                            function_args = {}

                    if function_name in self.tool_registry:
                        function_to_call = self.tool_registry[function_name]
                        try:
                            executed_output = function_to_call(**function_args)
                            tool_output_content = (
                                json.dumps(executed_output)
                                if isinstance(executed_output, (dict, list))
                                else str(executed_output)
                            )
                        except Exception as e:
                            tool_output_content = f"Error executing '{function_name}': {str(e)}"

                        print(
                            f"[Tool Execution] {function_name}({function_args}) -> {tool_output_content}"
                        )
                    else:
                        tool_output_content = f"Error: Tool '{function_name}' not found."

                    # Append the tool's output back into the message history
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": tool_output_content,
                        }
                    )
            else:
                # No more tool calls; return the final text response
                return response_message.content

        return "Error: Maximum tool-calling iterations reached without a final answer."