# flake8: noqa
PREFIX = """Answer the following questions as best you can.Please think in Japanese.
When searching, try the search query in English or Japanese, whichever you can find the best answer.
But please be sure to answer question in Japanese. You have access to the following tools"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""
