from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory

load_dotenv()
chat = ChatOpenAI()

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
    # chat_memory=FileChatMessageHistory("messages.json"),
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    content = input(">>: ")
    result = chain({"content": content})
    print(result["text"])
