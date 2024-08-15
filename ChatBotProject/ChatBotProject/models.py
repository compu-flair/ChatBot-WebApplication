from django.db import models



class Conversation(models.Model):
    time = models.DateTimeField(auto_now_add=True)
    user_id = models.CharField(max_length=25, default="1")

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    time = models.DateTimeField(auto_now_add=True)
    sender = models.CharField(max_length=100)
    text = models.TextField()


