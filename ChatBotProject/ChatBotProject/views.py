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




@csrf_exempt # cross site requests forgery attacks. 
def api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_msg = data["user_msg"]
        ## To do:
        # 1. retrieve docs
        # 2. make chain with memory
        # 3. ai_msg = chain.invoke(user_msg)
        ai_msg = llm.invoke(user_msg) ## this is temporary
        dic = {"api_response": ai_msg.content}
        return JsonResponse(dic)

    dic = {"api_response": "I cannot process GET requests. Please send a post request!"}
    return JsonResponse(dic)