# import os
# from dotenv import load_dotenv
# load_dotenv()

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


### system setting ###

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_ENV = os.getenv('PINECONE_ENV')

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
)

embeddings = OpenAIEmbeddings()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Answer the question always in Korean as if you are a very gentle and kind teacher, using ONLY the following context If you don't know the answer, just say you don't know. DON'T make anything up.
     
     context: {context}
     """),
    ("human", "{question}")
])

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
vectorstore = Pinecone.from_existing_index(index_name="demo", embedding=embeddings)
retriever = vectorstore.as_retriever()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message":message, "role":role})
        
def paint_history():
    for messages in st.session_state["messages"]:
        send_message(messages["message"], messages["role"], save=False)
        
# def get_history():
#     history = []
#     for messages in st.session_state["messages"]:
#         history.append(','.join([messages['message'], messages['role']]))
#     return ('/'.join(history))
        
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

  
### display setting ###

st.set_page_config(
    page_title="Senior GPT",
    page_icon="🚀"
)

st.title("Senior GPT")

st.markdown("""
            **Applied Text.** 크레버스 진학 핵심 용어사전 16
            """)

paint_history()

msg = st.chat_input("질문을 입력하세요")

if msg:
    send_message(msg, "human")

    chain = {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    } | prompt | llm
    
    response = chain.invoke(msg)
    send_message(response.content, "ai")
    