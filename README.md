# Sherlock-ML-for-text-labeling
Framework for machine learning projects at Insight Data Science.

- **Sherlock-ML-for-text-labeling** : 
    Given a dataset with limited labels and lots of unlabeled dataset,
    identify a smaller sample for labeling from the larger dataset

    Approach:
    Build a model using a smaller dataset. 
    Make predictions on the larger dataset using the model
    Pick the samples that are hard for model to predict for human labeling
   
    Dataset:
    Huffington post news articles with short description of the news item 
    It has 50 categories, out of which 10 major categories are chosen
  
## Setup
    data requirements:
    glove data available at https://nlp.stanford.edu/projects/glove/
    news article dataset available at https://www.kaggle.com/rmisra/news-category-dataset
    
## Source code src/
    
    src/pick_samples_from_semi_supervised.py loads the data, 
    builds a lstm model using text of the news articles embedded using glove
    picks the samples with low probabilities for human lableling
    compares the model built using picked samples that are difficult for model to predict 
    and model built from random sample
    
    

Clone repository and update python path
```
