# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt_template = """Use the following pieces of context to answer the question at the end. 
Be sure to answer the question in Japanese, not English. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
from datetime import datetime
import pytz

jst = pytz.timezone('Asia/Tokyo')
# 現在の日付と時刻を取得
datetime_jst = datetime.now(jst)
now = datetime_jst
now_of_year = now.strftime("%Y")
now_of_month = now.strftime("%m")
now_of_day = now.strftime("%d")
now_of_time = now.strftime("%H:%M")

system_template = """Today is the year {now_of_year}, the month is {now_of_month} and the date {now_of_day}. 
The current time is {now_of_time}.
You are a bot joining a Discord server where people are enjoying an online game called Dead by Daylight. 
You enjoy talking about Dead by Daylight with other users. If a user asks you a question, 
please use the following pieces of context to answer the users question. 
Be sure to answer the question in Japanese, not English.
If you don't know the answer, just say that you don't know, 
don't try to make up an answer.

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
