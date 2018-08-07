from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
import gensim
from gensim.models import KeyedVectors
from django.template.response import TemplateResponse
from django.shortcuts import render
from django.template import RequestContext
from glove import Corpus, Glove
from gensim.scripts.glove2word2vec import glove2word2vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity 
import mpld3
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cgi
import spacy
import numpy as np
from pymagnitude import *
from .models import gameData

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
            'firstSim':["gold"]}) 

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

    if phrase is not None:
        
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit = 200000)
         # gets rid of spaces in words to get interpreted by word2vec model
        phrase = ridSpace(phrase)

        try: 
            # word2vec code
            result4 = model.most_similar_cosmul(positive=[phrase, "cool"], negative=["adidas"],topn=number2)
        except KeyError:
            resultPhrase2 = "INPUT ERROR"
            resultPhrase3 = None

            return render(request, 'reasoning.html', {'resultPhrase2':[resultPhrase2],'first':["man"], 
                'second':["king"], 'third':["woman"], 'similarity':["boy"], 'phrase':phrase, 'number': [2], 
                'number2': [number2], 'secondSim':["silver"], 'firstSim':["gold"]}) 
     
        words2 = np.array(model.index2word[:10000])

        # load the spacy parser and part of speech tagger
        nlp = spacy.load('en')

        # ^^probably takes hella time

        words_pos = []
        proper = []
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

        component = model.get_vector(phrase)
        resultPhrase2 = get_similar(component, pos='ADJ', exclude_proper=True, top=number2)

        # sends results back to frontend
        return render(request, 'reasoning.html', {'resultPhrase2':resultPhrase2,'first':["man"], 'second':["king"], 
            'third':["woman"], 'similarity':["boy"], 'phrase':phrase, 'number': [2], 'number2': [number2], 
            'secondSim':["silver"], 'firstSim':["gold"]}) 

    # if similarity button is pressed
    if firstSim is not None:

        # gets rid of spaces in words to get interpreted by word2vec model
        firstSim = ridSpace(firstSim)
        secondSim = ridSpace(secondSim)

        try:
            # word2vec code for cosine similarity between two words
            result5Temp = model.similarity(firstSim, secondSim)
        except KeyError:
            result5 = "INPUT ERROR"

             # sends results back to frontend
            return render(request, 'reasoning.html', {'result5':[result5], 'first':["man"], 'second':["king"], 
                'third':["woman"], 'similarity':["boy"], 'number': [2], 'number2': [2], 
                'secondSim':[secondSim], 'firstSim':[firstSim]})


        result5 = str(round(result5Temp, 8))

        # sends results back to frontend with default values in other places
        return render(request, 'reasoning.html', {'result5':[result5], 'first':["man"], 'second':["king"], 
            'third':["woman"], 'similarity':["boy"], 'number': [2], 'number2': [2], 
            'secondSim':[secondSim], 'firstSim':[firstSim]})


    # sends results back to frontend with default values in other places
    return render(request, 'reasoning.html', {'first':["man"], 'second':["king"], 'third':["woman"], 
        'similarity':["boy"], 'number': [2], 'number2': [2], 'secondSim':["silver"], 
        'firstSim':["gold"]})

def graph(request):
    

    return render(request, 'graph.html')


