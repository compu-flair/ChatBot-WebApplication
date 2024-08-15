from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt # cross site requests forgery attacks. 
def api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_msg = data["user_msg"]
        ## ai_msg = llm.invoke(user_msg)
        ai_msg = "this is a temporary response from AI"
        dic = {"api_response": ai_msg}
        return JsonResponse(dic)

    dic = {"api_response": "I cannot process GET requests. Please send a post request!"}
    return JsonResponse(dic)