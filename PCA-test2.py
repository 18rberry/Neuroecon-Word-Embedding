import numpy as np
import pandas as pd
import gensim
import gensim.models.keyedvectors as word2vec
from pprint import pprint
from scipy import stats
import random
import matplotlib.pyplot as plt
import spacy
from sklearn.decomposition import FastICA, PCA
from sklearn.manifold import TSNE 


print("hello")



vec_path = '~/Downloads/GoogleNews-vectors-negative300.bin'

#original: word_vecs = word2vec.Word2Vec.load_w                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             ord2vec_format(vec_path, binary=True)
word_vecs = word2vec.KeyedVectors.load_word2vec_format(vec_path, binary=True)
# print(word_vecs["taco"] + word_vecs["bell"])

# print(word_vecs['McDonald's'])
# print(word_vecs['king'] + word_vecs['woman'] - word_vecs['man'])
# print(word_vecs.most_similar(positive=['woman', 'king'], negative=['man']))

#Most similar to king + woman - man
#[('queen', 0.7118192911148071), ('monarch', 0.6189674139022827), ('princess', 0.5902431011199951), ('crown_prince', 0.5499460697174072), ('prince', 0.5377321243286133), ('kings', 0.5236844420433044), ('Queen_Consort', 0.5235945582389832), ('queens', 0.518113374710083), ('sultan', 0.5098593235015869), ('monarchy', 0.5087411999702454)]




# print(word_vecs.most_similar(positive=['McDonalds']))
#[('Burger_King', 0.714678168296814), ('McDonalds_Burger_King', 0.679364025592804), ('McDonald', 0.6691861152648926), ('Mcdonalds', 0.6463509798049927), ('McDonalds_KFC', 0.6389850378036499), ('Kentucky_Fried_Chicken', 0.6374893188476562), ('Subway', 0.6350960731506348), ('Pizza_Hut', 0.6306871175765991), ('Mc_Donalds', 0.6301750540733337), ('Hardees', 0.6115859746932983)]

# print(word_vecs.most_similar(positive=['Chipotle']))
#[('Chipotle_Mexican_Grill', 0.7283411026000977), ('Panera', 0.6434450745582581), ('Qdoba', 0.6409419775009155), ('Cheesecake_Factory', 0.6365489363670349), ('PF_Chang', 0.6255013346672058), ('Taco_Bell', 0.608887791633606), ('Qdoba_Mexican_Grill', 0.6041489839553833), ('Whole_Foods', 0.6032172441482544), ('Texas_Roadhouse', 0.5914109349250793), ('Quiznos', 0.5906069874763489)]
#Check w/ glove
#CONSIDER NOUNS(?) after

# print(word_vecs["Taco_Bell"])



fastfood = ["McDonalds", "Burger King", "KFC","Subway","Taco Bell","Dairy Queen","Dunkin Donuts","Starbucks","Domino Pizza","Panera Bread","Arby","Chipotle","Pizza Hut","Chick-fil-a","Hardee","Popeyes","Whataburger","Quiznos","Zaxby","Steak-n-Shake","Qdoba","Panda Express","Del Taco","Krispy Kreme","Baskin Robbins"]

test = ["McDonalds", "Taco_Bell", "Chipotle", "Panera"]


result = []
dict_result = {}
for i in test:
    dict_result[i] = word_vecs[i]
    result.append(dict_result[i])

print(dict_result['McDonalds'])

# takes an array w/ list of brands
# returns a dictionary w/ labels of brands that were found in word2vec and corresponding vectors:
# if word not found, the function will leave it out of the array
def vector_array(brand_vectors):
    result = {}
    for i in fastfood:
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

#ff_dict: a dictionary:
#keys: fast food brands
#values: their word embeddings/vectors
#note: only brands that were found in the pretrained models
ff_dict = vector_array(fastfood)
print("what")
print(ff_dict['McDonalds'])


#@output: gives us the list of vectors we want to turn into a numpy array
def extract_list(dictionary):
    list = []
    for word, vector in dictionary.items():
        list.append(vector)
    return list

#ff_array: this is a numpy array of fastfood word embeddings/ vectors
ff_array = np.array(extract_list(ff_dict))
#should be Burger_King
print(ff_array[1])

#next steps: use PCA on a numpy array
#get the values of components 1 and 2 from PCA

#
#
pca = PCA(n_components= 2)
pca_ff_matrix = pca.fit_transform(ff_array)
pca_ff_matrix[1]
# nlp = spacy.load('en')

pca_ff_df = pd.DataFrame(data = pca_ff_matrix
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([pca_ff_df, axis = 1)


%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#scatterplot of fastfood, no labels yet
plt.figure(figsize=(8,8))
plt.scatter(pca_ff_matrix[:,0], pca_ff_matrix[:,1],
            c='black', edgecolor='', alpha=0.5)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)



#QUESTION: how do we improve the run time if we change the layout from static to dynamic?