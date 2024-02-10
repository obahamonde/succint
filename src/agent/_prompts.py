RUN_TEMPLATE: str = """
		<s> [INST]
		Based on user input determine if a tool call is gonna be performed. These are the tools that are available for this step: {{ definitions }}
		[/INST]
		The user input is:
		{{ message }}
		</s>
		
		[INST]
		If a tool call can be inferred according to Json Schema directly send a valid json object without any additional content.
		Don't include any kind of introductory comment or syntax no backticks (```) just the valid json object that was inferred in the following format:
		{ "tool": { "name": "tool_name", "parameters": { "parameter_name": "parameter_value" } } }
		
		[/INST]
		"""

CHAT_TEMPLATE: str = """
Similar results from knowledge base:

{{ context }}
"""
