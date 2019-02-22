# Word Embeddings Site
Website for interacting with word embedding models

# Setup info
You need to download magnitude file for keyvectors. 
Magnitude caches and allows keyvectors to loaded a lot faster after a lot of inputs. This is superior to using the regular Google News or Twitter KeyedVectors.
The link to download magnitude files and explanation: https://github.com/plasticityai/magnitude
Django will look for this file in the project root as ``GoogleNews-vectors-negative300.magnitude``.
