import datetime

from django.db import models
from django.utils import timezone


class gameData(models.Model):
    xValue = models.CharField(max_length=200)
    yValue = models.CharField(max_length=200)
    name = models.CharField(max_length=200)
    meanError = models.CharField(max_length=200)
