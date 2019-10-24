# Project 3 - Reddit Classification Problem

## Executive Summary

### Problem Statement

The objective is to pick two subreddits from Reddit and train a machine learning model that will be able to classify new posts into the correct subreddit. 

I identified r/whisky and r/wine as my two topics of interest because the two topics (i.e. based on real-world knowledge) share enough similarities to provide a good challenge, but are also differentiated enough that it should be possible to train a machine learning model. 

**Examples of simlarities**
* Both topics are part of the alcohol industry
* Flavour profile vocabulary e.g. rich, lush, citrus, toasty, full-bodied, length, finish
* Production jargon e.g. yeast, ferment, years, oak, stainless steel
* Periphery jargon e.g. bottle, collection, region, drink, flight, ABV

**Examples of differences**
* Ingredient terms e.g. rye, grain, malt vs. grapes, pinot noir, chardonnay
* Flavour profile vocabulary e.g. smoky, peaty vs. tannins
* Geographical regions e.g. Scotland, Islay, Kentucky, vs Australia, Burgundy, 
* Periphery terms e.g. dram, age, distillery, cask vs glass, vintage, winery, barrel

### Business relevance

**Search Engine Optimization**
Wine and whisky are the kinds of luxury class products that can feel intimidating because the domains are often thought of as requiring a lot of specialised knowledge and vocabulary. And while such jargon certainly show up a lot in brochures and specialist magazines, the reality is that the terms people use to talk about whiskies/wines in casual conversation tends to be more down-to-earth, comprehensible. And now with drinks companies also increasingly looking at e-commerce, knowing the everyday words that people use to discuss or search about whiskies/wines can help these companies improve their SEO game. 

**Email filtering**
For some reason, many smaller retailers in the alcohol sector are either very resistant against or slow in launching full-fledge e-commerce sites. Instead, their idea of retailing online is often uploading the product catalogue online in PDF form, such that prospective customers will have to email them to make enquiries or place purchase orders. This in turn means that when an email lands in the company's enquiries/sales inbox, someone will have to direct the email to the correct category manager. A good classification system would therefore help to automate and reduce much of this grunt work. 


### How it was carried out

I wanted to take an iterative approach and refine the quality of data cleaning/processing/modeling in stages. The benefits of this approach are two-fold
* I'll be able to get to my 'prototype' code more quickly - instead of being stuck in debugging! - by doing just very basic data cleaning/processing and modeling with default hyper-parameters.
* The process of putting together the prototype also allows me to take note of areas for improvement for the next round. 

#### Data Gathering
Thankfully, reddit has an API that's easy to work with, so I've tried to scrape 1,000 rows (maximum allowed via the API) of data (i.e. reddit posts) from the two subreddits, r/whisky and r/wine.

By default, the API returns about 25 rows (the number varies slightly in practice, as noted in the technical report) for each call, which are provided in the form of a JSON file. Since I didn't want to get my IP/MAC banned for overwhelming the system, a time.sleep() function was used to introduce a pause to stagger API calls.

The JSON file structure is akin to that of a Russian doll with many nested levels of dictionaries and lists, so I needed to explore the JSON file a fair bit to determine how to access the data I needed.

The extracted data was saved into separate csv files for each subreddit.

#### Data Cleaning and EDA

After the first round of data gathering, I realised that there was quite a number of duplicate rows for r/wine. It seems like if I'm trying to extract more rows that actually available, the API would just loop back to the top and return duplicate rows. Because of this, I ended up dropping 77 rows from r/wine. Since I'm still left with over 900 rows for r/wine, I felt the data set was large enough for training the machine learning model.

I also found out that a good 500+ rows in each subreddit had null values in the `selftext` (i.e. body text) field. However, since `title` was necessary, I decided I would concatenate `title` and `selftext` ínto `new_text` in the Pre-processing stage. But first, those null values had to be changed into empty strings - otherwise, concatenating a string and NaN would give me NaN.

#### Pre-processing

Creating the `new_text` column was the easy bit. There was a lot more cleaning up to do.

BeautifulSoup was helpful in extracting our text with minimal html tags, but it wasn't enough. Punctuation and things like `\n` and html links (without the helpful presence of a href tags!) were still left behind, so some extensive Regex work was needed to clean that out.

There were two other complications. 
* Emoji. Emoji is a big part of the way people communicate in casual settings, so we need to clean that out from the data set.
* Accented characters. With r/wine especially, I anticipate a lot of accented characters from things like French/Italian geographical regions and winery names. Since these accented words, e.g. 'château' will probably be a good predictor for the classifier, I wanted to make sure the words are retained - and in a legible form - so I had to decode them into UTF-8.

Numerals and stop words were also removed, and all text converted to lower case. 

#### Modeling

As mentioned above, I decided to take an iterative approach to modeling so that I can get to a working prototype quickly without being too bogged down in debugging or lengthy Grid Search runs. Besides, the defaults are there for a reason - because they are good-enough hyper-parameters for a broad range of cases.

For the first round, I wanted to achieve an overview of how CountVectorizer and TF-IDF vectorizing methods work, and also pit Logistic Regression against Multinomial Naive Bayes. The multinomial form of Naive Bayes was used because we are dealing with a multi-class, multi-experiment problem, and we are using word frequency (discrete) to determine our predictors. Default parameters were used for all components, although CountVectorizer and TF-IDF had the English `stop_words` specified. 

X_train and X_val scores were tabulated across all rounds, and I also included overfit value (i.e. X_train minus X_val) in the table to make it easier to tell which combination overfit the least.

#### Iteration 2

After doing the preliminary iteration, I had a better idea of the things I want to try to refine my model. 

**EDA and Pre-processing**

* Use a more sophisticated package i.e. SpaCy to lemmatize and tokenize our text - I felt that lemmatization would help to consolidate predictors better.
* Use GridSeach to tune CountVectorizer and TF-IDF on
    * `max_features`
    * `min_df` (remember, we don't want to set `max_df` so that we can preserve key predictors such as "wine" and "whisky"
    * `ngram_range`
* Compare subreddits using word clouds
   
**Modeling** 

* Use GridSearch to hypertune our hyper-parameters
* Compare a few classification models, i.e. Logistic Regression, Multinomial Naive Bayes, Random Forest Classifier

#### Comparing hyper-tuned models

Even after hyper-tuning, all models still had an overfitting problem, but I found out that using my hyper-tuned Multinomial Naive Bayes classifier with CountVectorizer gave me the best model (least amount of overfitting). 

CountVectorizer hyper-parameters:
````
* cvec__max_features': 1500
* 'cvec__min_df': 2 
* 'cvec__ngram_range': (1, 1)
````

Multinomial Naive Bayes hyper-parameters:
````
* 'alpha': 0.5
* 'fit_prior': False
````

* With CVEC, hyper-tuned LR and MNB tend to overfit to a smaller extent than their respective models with default hyper-parameters.
We might also be able to infer that reducing our features (by setting max_depth) for CVEC has also helped to reduce overfitting.
* However with TF-IDF, hyper-tuned models don't necessarily do better than models with default hyper-parameters.
* With CVEC, our hyper-tuned MNB was the best model giving us the highest X_val score, as well as the smallest overfit value.
* With TF-IDF, the results are again quite mixed. For instance, LR_def gave us the highest X_val score but it was LR_tuned that gave us the smallest overfit value.
* Overall, MNB_tuned provided the best results: the CVEC_X_val score was the highest of all X_val scores, and the corresponding CVEC_overfit_diff score was also the smallest in the table.

The most surprising thing from running Grid Search to hyper-tune my models was that I was recommended `max_depth: None` for Random Forest Classifier (the options I tried were None to 5) instead of a low integer as anticipated. Because of that, Random Forest Classifier was the most overfitting of all my models with a 'perfect' 1 on my X_train, but a whopping 0.1 lower accuracy on X_val. 

#### Evaluating the model with test set

To build my test data set, I did another round of API calls, attempting to retrieve 100 rows of data per subreddit. Next, I compared `name` (unique id for each post) between the test set and my train set, and dropped all duplicate from my test set. 

Unfortunately, r/wine was far more active than r/whisky so I ended up with 101 completely new rows of data for r/wine, and only 13 that were completey new for r/whisky. But for the test phase, it's perfectly fine not to have a balanced data set, so that was still okay. I also noted the baseline proportion of wine and whisky posts.

Using my best model (i.e. hyper-tuned Multinomial Naive Bayes classifier with CountVectorizer), I achieved an accuracy score of 0.886 on my test data set - not wonderful, but not too shabby either. 


#### Moving forward 

With supervised machine learning, there's always something else that can be tweaked and tuned. There's more that I would have liked to explore, but couldn't due to time constraints. Nonetheless, these are the areas I hope to able to look into

* Re-do the pre-processing process, this time without removing all hyperlinks via Regex. These days, most URLs are quite 'readable' (as opposed to random numbers) and tokens from URLs within `selftext` might be helpful in classification. For instance, the link `https://vinepair.com/articles/natural-wine-takes-australia/` would give us tokens 'vinepair', 'articles', 'natural', 'wine', 'takes', 'australia' and some of those tokens are intuitively good predictors (e.g. 'natural wine' is a big trend in wine but the word 'natural' doesn't come up much in whisky jargon).
* Add 'wine' and 'whisky' into the list of stop words to see how well the model can predict without these key predictors.
* Create my own list of stop words, to remove not-useful words such as 'bottle', 'taste'. Given that TF-IDF penalizes words with high frequencies, this might not matter. Cvec does not penalize these words by default, and I've also avoided setting `max_depth` for cvec, but setting a high `max_depth` might also serve to remove these words.
* Hyper-tune Random Forest Classifier again without 'None' as an option to see if it reduces overfitting. 


### Conclusion and Recommendations

