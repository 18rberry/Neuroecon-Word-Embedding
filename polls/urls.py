from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('reasoning', views.reasoning, name='reasoning'),
    path('graph', views.graph, name='graph'),
    path('api/graph', views.graph_api, name='graph_api'),
    path('api/analogy', views.analogy_api, name='analogy_api'),
    path('api/cosine_similarity', views.cosine_similarity_api, name='cosine_similarity_api'),
    path('api/most_similar', views.most_similar_api, name='most_similar_api'),
    path('api/adjectives', views.adjectives_api, name='adjectives_api'),
    path('api/vectors', views.vectors_api, name='vectors_api')
]
