from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('reasoning', views.reasoning, name='reasoning'),
    path('graph', views.graph, name='graph'),
    path('api/graph', views.graph_api, name='graph_api')
]
