# Word Embeddings Site
Website for interacting with word embedding models

![Example visualization](https://i.imgur.com/FM5AiYi.png)

Performs principle component analysis on corresponding word vectors for user input and plots along PC1 and PC2.

![Principle component descriptive words](https://i.imgur.com/FFl7QGQ.png)

Extracts descriptive adjectives for first four principle components.

# Setup info
You need to download magnitude file for keyvectors. 
Magnitude caches and allows keyvectors to loaded a lot faster after a lot of inputs. This is superior to using the regular Google News or Twitter KeyedVectors.
The link to download magnitude files and explanation: https://github.com/plasticityai/magnitude
Django will look for this file in the project root as ``GoogleNews-vectors-negative300.magnitude``.

# TODO
- Fix the graph animations on axis change to have silky smooth animations~
- Add ability to download data
- Separate API endpoint for labeling PCA components from the graphing part (Labeling the PCA components is generally pretty time consuming, people who just want to look at the graph might not care)
- Add support for multiple models (probably more .magnitude files so we can reuse existing code)
- Fix up the front page with less filler text
- tSNE support
