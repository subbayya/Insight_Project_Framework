# Sherlock-ML-for-text-labelin
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
Clone repository and update python path
```
