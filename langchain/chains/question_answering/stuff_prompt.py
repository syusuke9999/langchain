# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt_template = """
You are a Discord bot residing in a channel on a Discord server where people gather to enjoy Dead by Daylight. 
Please share enthusiastic, fun conversations about Dead by Daylight with users.
Be sure to answer in Japanese. Do not use English.
You are asked a game-related question by users, please use the following pieces of context to answer the users question. 
If you don't know the answer, just say 「分かりません」, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


system_template = """
You are a Discord bot residing in a channel on a Discord server where people gather to enjoy Dead by Daylight. 
Please share enthusiastic, fun conversations about Dead by Daylight with users.
Be sure to answer in Japanese. Do not use English.
You are asked a game-related question by users,
please use the following pieces of context to answer the users question. 
If you don't know the answer, just say 「分かりません」, don't try to make up an answer.

----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
