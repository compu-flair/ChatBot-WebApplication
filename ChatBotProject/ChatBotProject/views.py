from django.http import JsonResponse


def api(request):
    dic = {"api_response": "Hello World!"}
    return JsonResponse(dic)