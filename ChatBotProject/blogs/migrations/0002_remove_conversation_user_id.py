# Generated by Django 5.1 on 2024-08-16 00:49

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('blogs', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='conversation',
            name='user_id',
        ),
    ]
