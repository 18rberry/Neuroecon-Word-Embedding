from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
import gensim
from gensim.models import KeyedVectors
import gensim.models.keyedvectors as word2vec
from scipy import stats
import random
from django.template.response import TemplateResponse
from django.shortcuts import render
from django.template import RequestContext
# from glove import Corpus, Glove
# from gensim.scripts.glove2word2vec import glove2word2vec

#include below 2 lines for runtime error/ backend
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA, PCA
from sklearn.metrics.pairwise import cosine_similarity 
import mpld3
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cgi
import spacy
import numpy as np
from pymagnitude import *
from .models import gameData
from django.utils.safestring import mark_safe
from django.template import Library
import json


register = Library()


@register.filter(is_safe=True)
def js(obj):
    return mark_safe(json.dumps(obj))

def ridSpace(z):
    tempList = z.split()
    tempString = ""
    for c in tempList:
        q = tempList.pop(0)
        if(len(tempList)== 0):
            return q
        r = tempList.pop(0)
        tempString = q+"_"+r
        if(len(tempList)== 0):
            return tempString
        z = tempList.pop(0)
        tempString = tempString + "_" + z
    return tempString


def phraseReturn(x):
    z = x.split()
    r = ""
    for c in z:
        r = c + r
    return r


def index(request):
    

    return render(request, 'index.html')


def reasoning(request):

    # load the google word2vec model
    #model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit = 400000)
    model = Magnitude("GoogleNews-vectors-negative300.magnitude")

    # retrieves input information from frontend
    first = request.POST.get("first")
    second = request.POST.get("second")
    third = request.POST.get("third")
    similarity = request.POST.get("similarity")
    phrase = request.POST.get("phrase")
    numberTemp = request.POST.get("number")
    number2Temp = request.POST.get("number2")
    firstSim = request.POST.get("firstSim")
    secondSim = request.POST.get("secondSim")


    group = request.POST.get("group")
    print(group)


    # converts string numbers to ints and sets default value to 2
    if(numberTemp is None):
        number = 2
    else:
        number = int(numberTemp)

    if(number2Temp is None):
        number2 = 2
    else:
        number2 = int(number2Temp)


    # if analogy button is pressed
    if first is not None:

        # gets rid of spaces in words to get interpreted by word2vec model
        first = ridSpace(first)
        second = ridSpace(second)
        third = ridSpace(third)

        # word2vec code for analogies
        try: 
            result = model.most_similar_cosmul(positive=[second, third], negative=[first], topn=1)
        except KeyError:
            x = "INPUT ERROR"
            return render(request, 'reasoning.html', {'result':[x], 'first':[first], 'second':[second], 
                'third':[third], 'similarity':["boy"], 'number': [2], 'number2': [2], 'secondSim':["silver"], 
                'firstSim':["gold"]}) 
        
        x = result[0][0]
        y = result[0][1]

        # sends results back to frontend
        return render(request, 'reasoning.html', {'result':[x], 'first':[first], 'second':[second], 
            'third':[third], 'similarity':["boy"], 'number': [2], 'number2': [2], 'secondSim':["silver"], 
            'firstSim':["gold"], 't': test1})

    # if Top-N similarity button is pressed
    if similarity is not None:
        

        # gets rid of spaces in words to get interpreted by word2vec model
        similarity = ridSpace(similarity)

        try:
            # word2vec code for most similar
            result = model.most_similar(similarity, topn=number)
        except KeyError:
            result2 = "INPUT ERROR"
            result3 = None

            return render(request, 'reasoning.html', {'result2':[result2], 'result3': result3, 'first':["man"], 
                'second':["king"], 'third':["woman"], 'similarity':[similarity], 'number': [number], 'number2': [2], 
                'secondSim':["silver"], 'firstSim':["gold"]}) 
        
        # set the total amount of results to "number"
        z = 1
        temp = str(round(result[0][1], 3))
        result2 = [result[0][0]]
        result3 = [temp]
        while(z < number):
            result2.append(result[z][0])
            result3.append(str(round(result[z][1], 3)))
            z = z+1

        # sends results back to frontend
        return render(request, 'reasoning.html', {'result2':result2, 'result3': result3, 'first':["man"], 
            'second':["king"], 'third':["woman"], 'similarity':[similarity], 'number': [number], 'number2': [2], 
            'secondSim':["silver"], 'firstSim':["gold"]})

    #if adjectives button is pressed
    if phrase is not None:

        model = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True, limit = 200000)
         # gets rid of spaces in words to get interpreted by word2vec model

        adj_brand_list = phrase.split("\r\n")

        phrase_list = []

        for i in range(0, len(adj_brand_list)):
            phrase_list.append(ridSpace(adj_brand_list[i]))

        # try:
        #     # word2vec code; result 4 is a test/ verification
        #     #this gives: NIKE + cool - adidas
        #     #[('athletic', .294...),...] format
        #     result4 = model.most_similar_cosmul(positive=[phrase, "cool"], negative=["adidas"],topn=number2)
        # except KeyError:
        #     resultPhrase2 = "INPUT ERROR"
        #     resultPhrase3 = None
        #
        #     return render(request, 'reasoning.html', {'resultPhrase2':[resultPhrase2],'first':["man"],
        #         'second':["king"], 'third':["woman"], 'similarity':["boy"], 'phrase':phrase, 'number': [2],
        #         'number2': [number2], 'secondSim':["silver"], 'firstSim':["gold"], 'test': phrase_list, 'test1': adj_brand_list})

        #top 10,000 most common words
        words2 = np.array(model.index2word[:10000])

        # load the spacy parser and part of speech tagger
        nlp = spacy.load('en', disable = ['textcat','ner'])

        # ^^probably takes hella time

        words_pos = []
        proper = []

        #iterating through the most common words
        for w in words2:
            w = str(w)
            if w != '_':
                w = w.replace('_', ' ')
            p = nlp(w)[-1].pos_
            words_pos.append(p)
            proper.append(w != w.lower())
        words_pos = np.array(words_pos)
        proper = np.array(proper)

        arr = model.syn0[:10000]
        arr = arr / np.linalg.norm(arr, axis=1)[:, np.newaxis]

        def get_similar(vec,pos=None, exclude_proper=False, top=20):
            vecs = arr
            ws = words2
            ps = proper
            if pos is not None:
                subset = (words_pos == pos)
                vecs = vecs[subset]
                ws = ws[subset]
                ps = ps[subset]
            if exclude_proper:
                vecs = vecs[~ps]
                ws = ws[~ps]
                ix = np.argsort(-vec.dot(vecs.T))[:top]
            return ws[ix]

        final_output_list = []
        index_list = []

        for i in range(0, len(phrase_list)):
            index_list.append(i)
            component_i = model.get_vector(phrase_list[i])
            try:
                result_i = get_similar(component_i, pos='ADJ', exclude_proper=True, top=number2)
            except KeyError:
                result_i = "Input Not Found"
            final_output_list.append(result_i)


        # component = model.get_vector(phrase)
        # resultPhrase2 = get_similar(component, pos='ADJ', exclude_proper=True, top=nhttps://click.pstmrk.it/2sm/www.hackerrank.com%2Ftests%2Fdbsclk6pk50%2Flogin%3Fb%3DeyJ1c2VybmFtZSI6ImN6aHU0M0BiZXJrZWxleS5lZHUiLCJwYXNzd29yZCI6IjQ3OWUxNzBhIiwiaGlkZSI6dHJ1ZX0%3D/n-LA4AI/EDcI/_U1lbo-824/aHJ3LXRlc3QtaW52aXRlumber2)

        # sends results back to frontend
        return render(request, 'reasoning.html', {'resultPhrase2': final_output_list,'first':["man"], 'second':["king"],
            'third':["woman"], 'similarity':["boy"], 'phrase':phrase, 'number': [2], 'number2': [number2], 
            'secondSim':["silver"], 'firstSim':["gold"], 'test': phrase_list, 'test1': final_output_list})

    # if similarity button is pressed
    if firstSim is not None:

        result = []

        firstSim_list = firstSim.split("\r\n")
        secondSim_list = secondSim.split("\r\n")
        length_of_shortest_input = min(len(firstSim_list), len(secondSim_list))

        # gets rid of spaces in words (on the same line) to get interpreted by word2vec model
        for i in range(0, length_of_shortest_input):
            firstSim_list[i] = ridSpace(firstSim_list[i])
            secondSim_list[i] = ridSpace(secondSim_list[i])

        # gets rid of spaces in words to get interpreted by word2vec model
        firstSim1 = ridSpace(firstSim)
        secondSim = ridSpace(secondSim)

        for i in range(0, length_of_shortest_input):
            try:
                result5Temp = model.similarity(firstSim_list[i], secondSim_list[i])

            except KeyError:
                result5Temp = "Not Found"

            result.append(str(round(result5Temp, 8)))

        # sends results back to frontend with default values in other places
        return render(request, 'reasoning.html', {'result5': result, 'first':["man"], 'second':["king"],
            'third':["woman"], 'similarity':["boy"], 'number': [2], 'number2': [2],
            'secondSim':[secondSim], 'firstSim':[firstSim1], 'test': result})


    # sends results back to frontend with default values in other places
    return render(request, 'reasoning.html', {'first':["man"], 'second':["king"], 'third':["woman"], 
        'similarity':["boy"], 'number': [2], 'number2': [2], 'secondSim':["silver"], 
        'firstSim':["gold"]})





vec_path = '~/Downloads/GoogleNews-vectors-negative300.bin'
fastfood = ["McDonalds", "Burger King", "KFC","Subway","Taco Bell","Dairy Queen","Dunkin Donuts","Starbucks","Domino Pizza","Panera Bread","Arby","Chipotle","Pizza Hut","Chick-fil-a","Hardee","Popeyes","Whataburger","Quiznos","Zaxby","Steak-n-Shake","Qdoba","Panda Express","Del Taco","Krispy Kreme","Baskin Robbins"]
word_vecs = word2vec.KeyedVectors.load_word2vec_format(vec_path, binary=True)

def vector_array(brand_vectors):
    result = {}
    for i in brand_vectors:
        if ' ' in i:
            try:
                new = i.replace(" ", "_")
                result[i] = word_vecs[new]
            except:
                pass
        elif '-' in i:
            try:
                new = i.replace("-", "_")
                result[i] = (word_vecs[new])
            except:
                pass
        else:
            try:
                result[i] = word_vecs[i]
            except KeyError:
                pass

            #if brand name is something like "taco bell" or "Taco Bell": try "Taco_Bell"
            #this applies to two word brands only
    return result



def extract_list(dictionary):
    list = []
    for word, vector in dictionary.items():
        list.append(vector)
    return list

dict_test = {}
dict_test['hello'] = 12
dict_test['what'] = 25

def graph(request):
    # if statement required to load the page when no input has been typed in box
    # brands = (request.POST['brand_list_input']).split('\r\n')
    brands = ['hello', 'McDonalds']

    # brands is the list of brands and labels from user input
    if (request.POST.get('brand_list_input') != None):
        brands = (request.POST.get('brand_list_input')).split(" ")
    # for testing purposes
    else:
        brands = ["hello"]
    # single_wv is a dictionary whose keys are brands and values are 2 xy PCA coord lists'
    single_wv = {}
    master_dict = {}



    # # creates a dictionary brand_dict: brands, with their word vectors
    brand_dict = vector_array(brands)
    brand_array = np.array(extract_list(brand_dict))
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
#     single_wv = word_vecs[brands]
#     for number, label in enumerate(brands):
#         single_wv[number] = word_vecs[label]
#     return render(request, 'graph/result.html', {'single_wv': single_wv})