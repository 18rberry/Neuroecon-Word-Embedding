# Create your views here.
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.http import JsonResponse
from pymagnitude import *
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from itertools import chain
import numpy as np
import json
import pickle
import os

# Descriptive Adjectives caching
SKIP_CACHING = False
data_path = os.path.join('.', 'data')
vector_path = os.path.join(data_path, 'vectors')
model = Magnitude(
    os.path.join(vector_path, 'GoogleNews-vectors-negative300.magnitude')
)
adj_cache_path = os.path.join(data_path, 'adj_cache.pickle')
adj_list_path  = os.path.join(data_path, 'adjectives.txt')

model.query("warm") # warm up the model
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

def reformat(s):
    return '_'.join(s.replace('-', ' ').split()) if s else ''

def index(request):
    return render(request, 'index.html')


def safe_similarity(model, a, b):
    if a not in model:
        return '"{}" not found.'.format(a)
    if b not in model:
        return '"{}" not found.'.format(b)
    return str(round(model.similarity(a, b), 8))

def analogy_api(request):
    """
    Analogy API endpoint
    Params:
        pos1 - The first word in the complete analogy
        pos2 - The second word in the complete analogy
        neg1 - The first word in the incomplete analogy
    Returns:
        JSON with format:
        {
            'success': bool // Whether or the query succeeded
            'message': str  // Error message if success is false
            'result':  str  // The word that most closely fits the analogy 
        }
    """
    pos1 = reformat(request.GET.get('pos1', None))
    pos2 = reformat(request.GET.get('pos2', None))
    neg1 = reformat(request.GET.get('neg1', None))
    n    = int(request.GET.get('n', 1))
    if not all([pos1, pos2, neg1]):
        return JsonResponse({
            'success': False,
            'message': 'Missing input'
        })
    neg2 = model.most_similar_cosmul(
        positive=[pos1, pos2],
        negative=[neg1],
        topn=n
    )[0][0]
    result = {
        'success': True,
        'result': neg2
    }
    return JsonResponse(result)

def most_similar_api(request):
    raw_target = request.GET.get('target', None)
    target = reformat(raw_target)
    topn   = int(request.GET.get('topn', 10))
    if not target:
        return JsonResponse({
            'success': False,
            'message': 'Missing input'    
        })
    if target not in model:
        return JsonResponse({
            'success': False,
            'message': 'Target "{}" not found in the model'.format(raw_target)
        })
    result = []
    for word, sim in model.most_similar(target, topn=topn):
        result.append({
            'word': word,
            'similarity': float(sim)
        })

    return JsonResponse({
        'success': True,
        'result': result
    })

def cosine_similarity_api(request):
    list1 = list(map(reformat, request.GET.get('list1', '').split(',')))
    list2 = list(map(reformat, request.GET.get('list2', '').split(',')))
    if len(list1) != len(list2):
        return JsonResponse({
            'success': False,
            'message': 'Inputs are different lengths'
        })
    if not (list1 or list2):
        return JsonResponse({
            'success': False,
            'message': 'Missing input'
        })
    result = [
        {'word1': a, 'word2': b, 'similarity': safe_similarity(model, a, b)} 
        for a, b in zip(list1, list2)
    ]
    return JsonResponse({
        'success': True,
        'result': result
    })
    
def adjectives_api(request):
    raw_target = request.GET.get('target', None)
    target = reformat(raw_target)
    n      = int(request.GET.get('n', 10))
    if not target:
        return JsonResponse({
            'success': False,
            'message': 'Please input a target'
        })
    if target not in model:
        return JsonResponse({
            'success': False,
            'message': 'Target "{}" not found in the model'.format(raw_target)
        })
    v = model.query(target)
    result = [
        {'word': word, 'similarity': similarity}
        for word, similarity in get_descriptive_adjectives(v, n=n)
    ]
    return JsonResponse({
        'success': True,
        'result': result
    })
    
def vectors_api(request):
    raw_targets = request.GET.get('targets', '')
    if not raw_targets:
        return JsonResponse({
            'success': False,
            'message': 'Missing input'
        })
    rows = []
    for raw in raw_targets.split(','):
        row = OrderedDict()
        row['word'] = raw
        target = reformat(raw)
        if target not in model:
            row['found'] = False
            for i in range(300):
                row[i] = 0
        else:
            row['found'] = True
            for i, val in enumerate(model.query(target)):
                row[i] = float(val)
        rows.append(row)
    return JsonResponse({
        'success': True,
        'result': rows
    })
        


def reasoning(request):
    # retrieves input information from frontend
    analogy1    = reformat(request.POST.get('analogy1', 'man'))
    analogy2    = reformat(request.POST.get('analogy2', 'king'))
    analogy3    = reformat(request.POST.get('analogy3', 'woman'))
    topn_sim    = reformat(request.POST.get('topn_sim', 'boy'))
    adj_phrase  = reformat(request.POST.get('adj_phrase',   'Nike'))
    similarity1 = request.POST.get('similarity1', 'silver')
    similarity2 = request.POST.get('similarity2', 'gold')
    topn_count  = int(request.POST.get('topn_count', 10))
    adj_count   = int(request.POST.get('adj_count',  10))

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
        phrase_vec = model.query(adj_phrase)
        adj_results = map(
            lambda x: x[0],
            get_descriptive_adjectives(phrase_vec, n=adj_count)
        )
        return render(request, 'reasoning.html', locals())

    # Similarity
    if request_type == 'similarity':
        list1 = list(map(reformat, similarity1.splitlines()))
        list2 = list(map(reformat, similarity2.splitlines()))
        similarity_results = [
            round(model.similarity(a,b), 8) for a,b in zip(list1, list2)
        ]
        return render(request, 'reasoning.html', locals())

    # If regular page load, just pass plain defaults
    return render(request, 'reasoning.html', locals())

def get_descriptive_adjectives(v, n=30, select=lambda l,n: l[:n]):
    adj_list = list(map(
            lambda i: [i[0], float(cosine_similarity([i[1]], [v])[0][0])],
            adj_map.items()
    )) # Convert adj_map into list of tuples of (Adjective str, Cosine similarity)
    adj_list.sort(
        key = lambda a: a[1],
        reverse = True
    ) # Sort by cosine similarity
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
            #if brand name is something like 'taco bell' or 'Taco Bell': try 'Taco_Bell'
            #this applies to two word brands only
            #use itertools.product 
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
    result['axis_labels'] = axis_labels
    return JsonResponse(result)