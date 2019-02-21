from django.urls import path

from . import views

urlpatterns = [
    path('index', views.index, name='index'),
    path('reasoning', views.reasoning, name='reasoning'),
    path('graph', views.graph, name='graph'),
    path('result', views.result, name='result')
    # path('reasoningExtended', views.reasoningExtended, name='reasoningExtended'),
    # path('stereotype', views.stereotype, name='stereotype')
]

#when calling graph: uses GET method
#graph_form
#path('graph_form', views.graph_form, name = 'graph_form')

#http verbs:
#GET: retrieve remote data
#POST: create new data