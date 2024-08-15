from .models import Conversation, Message
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from dotenv import load_dotenv
load_dotenv(".env")
openai_api_key = os.getenv("openai_api_key")


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.)
# or
# from langchain_google_vertexai import ChatVertexAI
# llm = ChatVertexAI(model="gemini-pro")


from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large',openai_api_key=openai_api_key)

############################################################################################
## move this block to a separate function and run it weekly to always have updated site info
############################################################################################
from langchain_community.document_loaders.sitemap import SitemapLoader
sitemap_loader = SitemapLoader(web_path="https://apps.compu-flair.com/sitemap.xml")
documents = sitemap_loader.load()

############################################################################################
############################################################################################
## load documents from server database in production
# ??????

## temporary vector database. We need to replace this in production
from langchain_community.vectorstores import FAISS
vector = FAISS.from_documents(documents, embeddings)


from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.

Context:
{context}
                                          
Question: 
{question}
                                          
Your response:
                                          
""")

from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)


from langchain_core.messages import HumanMessage, AIMessage



from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

instruction_to_system = """
Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is.
"""

question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_to_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


question_chain = question_maker_prompt | llm | StrOutputParser()

## part of adding memory
def question_func(chat_history, question):
    if len(chat_history) == 0:
        return question
    else:
        new_question = question_chain.invoke({'question':question, 'chat_history':chat_history})
        print(f"new_question: {new_question}")
        return new_question

chain = setup_and_retrieval | prompt | llm | output_parser




@csrf_exempt # cross site requests forgery attacks. 
def api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_msg = data["user_msg"]
        user_id  = data["user_id"]
        ## To do:
        # 1. retrieve docs
        # 2. make chain with memory
        # 3. ai_msg = chain.invoke(user_msg)

        ############################
        conversation = Conversation.objects.get(user_id=user_id)
        messages = Message.objects.filter(conversation=conversation)
        chat_history = [] 
        for m in messages: ## add previous messages in the current conversation
            if m.sender == "user":
                chat_history.append(HumanMessage(content=m.text))
            else:
                chat_history.append(AIMessage(content=m.text))
        
        
        ai_msg = chain.invoke(question_func(chat_history, user_msg))
        Message.objects.create(
            conversation = conversation,
            sender = "user",
            text = user_msg
        )
        Message.objects.create(
            conversation = conversation,
            sender = "bot",
            text = ai_msg
        )
        ############################
        
        dic = {"api_response": ai_msg}
        return JsonResponse(dic)

    dic = {"api_response": "I cannot process GET requests. Please send a post request!"}
    return JsonResponse(dic)