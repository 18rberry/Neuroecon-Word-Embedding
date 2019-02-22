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
    first      = reformat(request.POST.get("first"))
    second     = reformat(request.POST.get("second"))
    third      = reformat(request.POST.get("third"))
    similarity = reformat(request.POST.get("similarity"))
    phrase     = reformat(request.POST.get("phrase"))
    firstSim   = request.POST.get("firstSim")
    secondSim  = request.POST.get("secondSim")

    # Handle number conversions
    number     = request.POST.get("number")
    number2    = request.POST.get("number2")
    number     = int(number)  if number  else 2
    number2    = int(number2) if number2 else 2

    # Default render params
    defaults = {
        'first'     : [first      or 'man'   ], 
        'second'    : [second     or 'king'  ], 
        'third'     : [third      or 'woman' ], 
        'similarity': [similarity or 'boy'   ], 
        'number'    : [number     or  2      ], #These or's aren't needed
        'number2'   : [number2    or  2      ], #But I wanted stuff to line up
        'firstSim'  : [firstSim   or 'gold'  ],
        'secondSim' : [secondSim  or 'silver'],
        'phrase'    : phrase     or 'Nike'     #Odd one out : (
    }

    # Analogies
    if first:
        result = model.most_similar_cosmul(
            positive=[second, third],
            negative=[first],
            topn=1
        )
        defaults['result'] = [result[0][0]]
        return render(request, 'reasoning.html', defaults)

    # Top-N Similar Words
    if similarity:
        result = model.most_similar(similarity, topn=number)
        defaults['result2'], defaults['result3'] = [], []
        for word, cos_sim in result:
            defaults['result2'].append(word)
            defaults['result3'].append(round(cos_sim, 3))
        return render(request, 'reasoning.html', defaults)

    # Adjectives
    if phrase:
        phrase_vec = model.query(phrase)
        adj_list = list(adj_map)
        adj_list.sort(
            key = lambda a: cosine_similarity(
                [adj_map[a]],
                [phrase_vec]
            )[0][0],
            reverse = True
        )
        defaults['resultPhrase2'] = adj_list[:number2]
        return render(request, 'reasoning.html', defaults)

    # Similarity
    if firstSim:
        list1 = list(map(reformat,  firstSim.splitlines()))
        list2 = list(map(reformat, secondSim.splitlines()))
        result = [
            round(model.similarity(a,b), 8) for a,b in zip(list1, list2)
        ]
        defaults['result5'] = result
        return render(request, 'reasoning.html', defaults)

    # If regular page load, just pass plain defaults
    return render(request, 'reasoning.html', defaults)

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
    # Debateably we should move all of this to an API
    brand_input = request.POST.get('brand_list_input')
    brands = brand_input.splitlines() if brand_input else ['hello', 'bye']

    # single_wv is a dictionary whose keys are brands and values are 2 xy PCA coord lists'
    single_wv = {}
    master_dict = {}

    # creates a dictionary brand_dict: brands, with their word vectors
    brands, brand_array = to_vector_dict(brands)
    pca = PCA(n_components=2)

    # pca matrix for 2 component PCA on list brands
    pca_matrix = pca.fit_transform(brand_array)

    variance = pca.explained_variance_ratio_
    try:
        variance1 = str((variance[0]*100).round(2)) + "%"
        variance2 = str((variance[1]*100).round(2)) + "%"
        vartotal = str(((variance[0] + variance[1])*100).round(2)) + "%"
        master_dict['vari1'] = variance1
        master_dict['vari2'] = variance2
        master_dict['vartot'] = vartotal
    except IndexError:
        pass

    for index, label in enumerate(brands):
        single_wv[label] = pca_matrix[index]
    # wv_list is a list
    for index, label in enumerate(single_wv):
        key = "x" + str(index)
        master_dict[key] = single_wv[label]

    master_dict['labs'] = brands
    return render(request, 'graph.html', master_dict)

def graph_api(request):
    labels = request.GET.get('labels', '').split(',')
    print(labels)
    n_components = int(request.GET.get('n_components', 2))
    labels, vectors = to_vector_dict(labels)
    if len(labels) < n_components:
        print('meow')
        print(n_components)
        return JsonResponse({'success': False})
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
        for word, vec in model.most_similar(axis, topn=20):
            label_dict[word] = vec
        axis_labels.append(label_dict)
    result["axis_labels"] = axis_labels
    from pprint import pprint
    pprint(result)
    return JsonResponse(result)
