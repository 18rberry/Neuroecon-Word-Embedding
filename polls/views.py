# Create your views here.
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.template import Library
from django.http import JsonResponse
from pymagnitude import *
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import numpy as np
import json
import pickle
import os

# Descriptive Adjectives caching
SKIP_CACHING = False
data_path = os.path.join('.', 'data')
vector_path = os.path.join(data_path, 'vectors')
model = Magnitude(
    os.path.join(vector_path, "GoogleNews-vectors-negative300.magnitude")
)
adj_cache_path = os.path.join(data_path, 'adj_cache.pickle')
adj_list_path  = os.path.join(data_path, 'adjectives.txt')

if SKIP_CACHING:
    print('Skipping adjective cache. Adjectives will be disabled.')
elif os.path.exists(adj_cache_path):
    with open(adj_cache_path, 'rb') as f:
        adj_map = pickle.load(f) 
else:
    print('Caching adjectives. This will take a few seconds...')
    with open(adj_list_path, 'rt') as f:
        adj_map = {}
        for i, adj in enumerate(f.read().splitlines()):
            adj_map[adj] = model.query(adj)
            sys.stdout.write('\r> {} adjectives processed.'.format(i))
            sys.stdout.flush()
    print('\nFinished caching!') 
    with open(adj_cache_path, 'wb') as f:
        pickle.dump(adj_map, f)

register = Library()

@register.filter(is_safe=True)
def js(obj):
    return mark_safe(json.dumps(obj))

def reformat(s):
    return '_'.join(s.replace('-', ' ').split()) if s else ''

def index(request):
    return render(request, 'index.html')

def reasoning(request):
    # retrieves input information from frontend
    analogy1 = reformat(request.POST.get("analogy1", 'man'))
    analogy2 = reformat(request.POST.get("analogy2", 'king'))
    analogy3 = reformat(request.POST.get("analogy3", 'woman'))
    topn_sim = reformat(request.POST.get("topn_sim", 'boy'))
    phrase     = reformat(request.POST.get("phrase", 'Nike'))
    similarity1 = request.POST.get("similarity1", 'silver')
    similarity2 = request.POST.get("similarity2", 'gold')
    topn_count = int(request.POST.get("topn_count", 10))
    adj_count  = int(request.POST.get("adj_count",  10))

    request_type = request.POST.get('type')
    # Handle number conversions

    # Analogies
    if request_type == 'analogy':
        result = model.most_similar_cosmul(
            positive=[analogy2, analogy3],
            negative=[analogy1],
            topn=1
        )
        analogy_result = result[0][0]
        return render(request, 'reasoning.html', locals())

    # Top-N Similar Words
    if request_type == 'topn_similar':
        result = model.most_similar(topn_sim, topn=topn_count)
        topn_words, topn_sims = [], []
        for word, cos_sim in result:
            topn_words.append(word)
            topn_sims.append(round(cos_sim, 3))
        return render(request, 'reasoning.html', locals())

    # Adjectives
    if request_type == 'adjectives':
        phrase_vec = model.query(phrase)
        adj_results = map(
            lambda x: x[0],
            get_descriptive_adjectives(phrase_vec, n=adj_count)
        )
        return render(request, 'reasoning.html', locals())

    # Similarity
    if request_type == 'similarity':
        list1 = list(map(reformat,  similarity1.splitlines()))
        list2 = list(map(reformat, similarity2.splitlines()))
        similarity_results = [
            round(model.similarity(a,b), 8) for a,b in zip(list1, list2)
        ]
        return render(request, 'reasoning.html', locals())

    # If regular page load, just pass plain defaults
    return render(request, 'reasoning.html', locals())

def get_descriptive_adjectives(v, n=30, select=lambda l,n: l[:n]):
    adj_list = list(map(
        lambda i: (i[0], cosine_similarity([i[1]], [v])[0][0]),
        adj_map.items()
    ))
    adj_list.sort(
        key = lambda a: a[1],
        reverse = True
    )
    return select(adj_list, n)

def axis_select(adj_list, n):
    top = adj_list[:n//2]
    bot = list(reversed(adj_list[-n//2:]))
    return top + [('-'*10,0)] + bot

def to_vector_dict(labels):
    result = OrderedDict()
    for label in labels:
        clean_label = reformat(label)
        if clean_label in model:
            result[label] = model.query(clean_label)
            #if brand name is something like "taco bell" or "Taco Bell": try "Taco_Bell"
            #this applies to two word brands only
    return list(result), list(result.values())


def graph(request):
    return render(request, 'graph.html')

def graph_api(request):
    labels = request.GET.get('labels', '').split(',')
    n_components = int(request.GET.get('n_components', 2))
    labels, vectors = to_vector_dict(labels)
    if len(labels) < n_components:
        return JsonResponse({'success': False})

    # Run PCA on vecs
    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(vectors)
    variance = pca.explained_variance_ratio_
    result = OrderedDict()
    result['success'] = True
    result['variance'] = list(variance)

    # Points
    points = OrderedDict()
    for i, label in enumerate(labels):
        points[label] = list(pca_matrix[i])
    result['points'] = points

    # Axis labels
    axis_labels = []
    for axis in pca.components_:
        label_dict = OrderedDict()
        for word, sim in get_descriptive_adjectives(
            axis, select=axis_select
        ):
            label_dict[word] = sim
        axis_labels.append(label_dict)
    result["axis_labels"] = axis_labels
    return JsonResponse(result)
