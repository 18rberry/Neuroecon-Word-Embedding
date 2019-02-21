# Create your views here.
from django.shortcuts import render
from django.utils.safestring import mark_safe
from django.template import Library
from pymagnitude import *
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import pickle
import os

# Descriptive Adjectives caching
SKIP_CACHING = False
model = Magnitude("GoogleNews-vectors-negative300.magnitude")
adj_cache_path = 'adj_cache.pickle'

if SKIP_CACHING:
    print('Skipping adjective cache. Adjectives will be disabled.')
elif os.path.exists(adj_cache_path):
    with open(adj_cache_path, 'rb') as f:
        adj_map = pickle.load(f) 
else:
    print('Caching adjectives. This will take a few seconds...')
    with open('adjectives.txt', 'rt') as f:
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
    return '_'.join(s.split()) if s else ''

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

def vector_array(brand_vectors):
    result = {}
    for i in brand_vectors:
        i = i.replace(" ", "_").replace('-', '_')
        if i in model:
            result[i] = model.query(i)
        else:
            result[i] = np.array([0] * 300)
            #if brand name is something like "taco bell" or "Taco Bell": try "Taco_Bell"
            #this applies to two word brands only
    return result



def extract_list(dictionary):
    list = []
    for word, vector in dictionary.items():
        list.append(vector)
    return list

def graph(request):
    # if statement required to load the page when no input has been typed in box
    # brands = (request.POST['brand_list_input']).split('\r\n')
    brands = ['hello', 'McDonalds', 'bye']

    # brands is the list of brands and labels from user input
    if (request.POST.get('brand_list_input') != None):
        brands = (request.POST.get('brand_list_input')).splitlines()
    # for testing purposes
    else:
        brands = ["hello", 'cya']
    # single_wv is a dictionary whose keys are brands and values are 2 xy PCA coord lists'
    single_wv = {}
    master_dict = {}

    # creates a dictionary brand_dict: brands, with their word vectors
    brand_dict = vector_array(brands)
    print(list(brand_dict))
    brand_array = np.array([v for k,v in brand_dict.items()])
    print(len(brand_array))
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

    for number, label in enumerate(brand_dict):
        single_wv[label] = pca_matrix[number]
    # wv_list is a list
    for index, label in enumerate(single_wv):
        key = "x" + str(index)
        master_dict[key] = single_wv[label]

    master_dict['labs'] = brands


    return render(request, 'graph.html', master_dict)

def result(request):

    #if statement required to load the page when no input has been typed in box
    # brands = (request.POST['brand_list_input']).split('\r\n')
    # brands = ['hello', 'McDonalds']

    #brands is the list of brands and labels from user input
    if (request.POST.get('brand_list_input') != None):
        brands = (request.POST.get('brand_list_input')).split(" ")
    #for testing purposes
    else:
        brands = ["hello"]
    single_wv = {}
    master_dict = {}



    #
    # # creates a dictionary brand_dict: brands, with their word vectors
    # brand_dict = vector_array(brands)
    # brand_array = np.array(extract_list(brand_dict))
    # pca = PCA(n_components=2)
    # # pca matrix for 2 component PCA on list brands
    # pca_matrix = pca.fit_transform(brand_array)
    #
    #
    #
    # # single_wv is a dictionary whose keys are brands and values are 2 xy PCA coord lists'
    # length_pca_matrix = len(pca_matrix)
    # length_single_wv = len(single_wv)
    # for number, label in enumerate(brand_dict):
    #     single_wv[label] = pca_matrix[number]
    # #wv_list is a list
    # for index, label in enumerate(single_wv):
    #     key = "x" + str(index)
    #     master_dict[key] = single_wv[label]
    #
    # master_dict['labs'] = brands
    #
    #
    #
    #
    #
    #
    # num_iters = len(brands)
    # new_dict = {}
    # # for i in num_iters:
    # #     new_dict[i] = single_wv[i]
    # # new_dict['tt'] = brands
    # # new_dict['tt2'] = "hello"
    # # list_test = [1, 2, .5]
    # # single_digit = 1.5
    # # value = pca_matrix[1][1]
    # #first n keys of master_dict = x0, x1 etc. whose values are xy coord pairs
    # #last key = "labs"
    #


    return render(request, 'result.html', master_dict)

    # return render(request, 'polls/graph.html', {'single_wv': single_wv})
    #
    # dict_test = {}
    # dict_test['hello'] = 12
    # dict_test['what'] = 25
    # label = {}
    # # brands = (request.POST['brand_list_input']).split('\r\n')
    # dict_test['list'] = ['Acura', 'Honda']
    # brands = request.POST.get('brand_list_input')
    # dict_test['yep'] = brands
    # return render(request, 'graph.html', dict_test)

#receive data from graph.html textbox
#use this function to apply PCA


# def result(request):
#     brands = request.POST['brand_list_input']
#     single_wv = model[brands]
#     for number, label in enumerate(brands):
#         single_wv[number] = model[label]
#     return render(request, 'graph/result.html', {'single_wv': single_wv})
