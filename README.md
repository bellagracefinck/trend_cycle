## The Devil Wears Prada, I Wear Whatever's on Clearance: NLP Applications in Fashion Trend Forecasting and Brand Longevity Analysis
### Bella Grace Finck, DATA 340

## Abstract

The concept of fashion trend forecasting has existed for centuries, but recent strides in natural language processing and AI have created countless opportunities to fortify the work done by forecasting agencies and consumers alike. Through a thorough analysis of Vogue Runway's online archive of Ready-to-Wear collections from the past 34 years, this project aims to investigate one question in particular: 

**How successful are natural language processing efforts at recognizing the trend cycle in action, identifying the characteristics of fashion designers who achieve long-term success, and using the two to predict trends and designer brand longevity alike?**

The project enlists a multifaceted approach involving Named Entity Recognition, network analysis, clustering algorithms, and classification models in order to explore the interplay between language and longevity in the world of high fashion. Results of these analyses reveal patterns of newer designers creating collections inspired by others, more established/prevalent designers utilizing timeless design characteristics rather than bold or unique ones, and the evolution of fashion trends over time. Tangentially, we analyze the structure of fashion news coverage over time, noting a particular increase in style descriptions in the 2020s with the rise of "aesthetic" culture and self-categorization. 


## Introduction
One of my favorite movie monologues of all time is by Meryl Streep (as fictional fashion magazine editor Miranda Priestly) in the 2006 cult classic The Devil Wears Prada. After a poorly timed joke by her extremely oblivious but well-meaning assistant, Streep famously exposes the trend cycle as we know it today while simultaneously decimating Anne Hathaway and shining a light on the true power of the fashion industry leaders in the process. (https://www.youtube.com/watch?v=vL-KQij0I8I)
       
> “You… go to your closet, and you select… I don’t know, that lumpy blue sweater, for instance, because you’re trying to tell the world that you take yourself too seriously to care about what you put on your back, but what you don’t know is that that sweater is not just blue, it’s not turquoise, it’s not lapis, it’s actually cerulean. You’re also blithely unaware of the fact that, in 2002, Oscar de la Renta did a collection of cerulean gowns, and then I think it was Yves Saint Laurent, wasn’t it?… who showed cerulean military jackets. […] And then cerulean quickly showed up in the collections of eight different designers. Then it filtered down through the department stores and then trickled on down into some tragic casual corner where you, no doubt, fished it out of some clearance bin. However, that blue represents millions of dollars of countless jobs, and it’s sort of comical how you think that you’ve made a choice that exempts you from the fashion industry when, in fact, you’re wearing a sweater that was selected for you by the people in this room… from a pile of ‘stuff.’” - *The Devil Wears Prada*, 2006
        
Even as I read the words for the hundredth time, I can feel the uneasy sensation I got the first time I ever watched the film. In a world in which the illusion of choice is everywhere, fashion is often advertised as an escape - an opportunity for one to make choices based exclusively on their own intended self-expression. However, just like most things in our society, the vast majority of decisions we make have already been made for us by a group of highly influential elites. 

I have always been interested in fashion trend forecasting and occasionally attempt to qualitatively analyze things myself, but this project is by far my most ambitious endeavor into the world of trend forecasting yet. As I began to research current methodologies in trend forecasting agencies, I discovered that most projects are aimed at evaluating the lifetimes of trends rather than evaluating the characteristics that enable the most successful members of the fashion industry to maintain their success and influence. This makes sense in the context of our efficiency and profit-driven world, but my instincts tell me that trend forecasting as a field would benefit from the greater contextualization of analyzing brand longevity. 

I hope that by discovering similarities between different designer collections and overall seasons I will be able to identify and anticipate potential upcoming trends and that by analyzing designer prevalence over time I will be able to predict the future success of specific designers. 


## Methodology/Dataset

#### Data collection
Data was scraped from the Vogue Runway archive using BeautifulSoup (see data_generation notebook). Although it originated as a print-only publication, Vogue has gradually shifted to include their articles and images in a digital format on their website. They have also begun to digitize collections from before the establishment of their digital presence, dating back to Fall 1988. My dataset is made up of every ready-to-wear collection in the archive since Spring 1990 and through Fall 2023. I have saved the raw article text for each of these collections along with the respective season, year, and designer in my complete.csv file.  

It is important to note that while this dataset is large, it is not exhaustive - particularly in the earlier seasons - due to the fact that Vogue is currently expanding the Vogue Runway archive by digitizing collections originally shot on film and reviews originally written for print.   

Given the aims of my project, I opted to use Ready to Wear collections rather than Couture. This was based on the assumption that RTW collections are truer to the products that designers are actually selling to a consumer base, therefore they are the collections that will influence the trend cycle the most. While all designer clothing is an art form, there tends to be more repetition and influence in RTW collections than in couture collections, in which there is very little practicality required of the clothing itself. 

#### Data format
Collection descriptions are located in the *collections* dataframe, which consists of the columns 'season', 'year', 'seasonyear', 'designer', 'text', 'id', and 'preprocessed_sentences' and is eventually mapped to include 'consistency', 'prevalence', and 'class'.

Designer-specific information is contained in the *designers* dataframe, which consists of 'designer', 'collections' (total number of designer's collections), and 'first_season', 'consistency', 'prevalence', and 'class'. 

In the preprocessing portion of my feature engineering, I implemented the StanfordNER model that identifies people's names in text. This was a necessity given how common it is for collection reviews to include the names of creative directors (which more often than not identify the designer) and models (which inadvertently date the collection). In order to ensure that the dataset was gleaned of any blatantly identifying characteristics, I removed all names and designer names from the collection descriptions in the preprocessing function. 

#### Metric calculations
Designers are quantified by two metrics: *consistency* and *prevalence*.  

- *consistency*: The total number of collections made by designer/total # of seasons since the designer’s initial season (inclusive of first season)
- *prevalence*: An adapted version of the consistency metric that penalizes designers who have few collections. This was put in place for two reasons - 1) to penalize designers with high consistency values as a result of having only been around for a short period of time (ex: a designer whose first collection was in the most recent season has a consistency value of 1.0) and 2) to further penalize designers who have low consistency and few collections. I created the prevalence formula using a **penalty term**, $α$.
    - α comes in several forms, listed here in order from least to most severe:
        1. $α = \frac{1}{collections^2}$
        2. $α = \frac{1}{collections}$
        3. $α = \frac{1}{\sqrt{collections}}$
        4. $α = \frac{1}{\sqrt[3]{collections}}$
    - This helps to ensure that designers who have only been in the most recent season (consistency = 1) are not weighted equally with designers who have high consistency values after having been around for many years

![Penalties](https://github.com/bellagracefinck/trend_cycle/blob/main/images/penalty.png)
In these plots, you can see the impact of the different penalty terms. In my final analysis, I used the third form of $α$, $α = \frac{1}{\sqrt{collections}}$, as it penalized the brand new designers with high consistency values without limiting high prevalence values to exclusively the oldest, most established designers. 

The prevalence metric is then calculated as $prevalence = consistency - \frac{1}{\sqrt{collections}}$.

The *class* variable is a class assignment by percentile of prevalence score. Classes range from 0 to 5 and the 0-20, 20-40th, 40-60th, 60-80th, 80-90th, and 90-94th, anf 95-100th percentiles, respectively. 

### Analysis techniques
#### Trend cycle analysis
In order to investigate the first portion of the research question - recognizing the trend cycle in real time - I decided to implement some principles of network analysis as well as traditional clustering algorithms in order to identify similar collections within the dataset. In order to extract the summarizing features of the collections, I created a custom Named Entity Recognition model that was trained specifically on fashion-related entities. 

The NER model searched for five different entity types:
- COLOR: colors, ranging in specificity from 'red' to 'russet'
- MATERIAL: fabric types, such as 'organza', 'tulle', or 'cotton'
- GARMENT_TYPE: different types of clothing, such as 'shirt', 'parka', or 'camisole'
- FEATURE: item features/descriptors, including different cuts ('sleeveless, 'tailored', 'mermaid'), textures ('padded', 'shiny'), and other miscellaneous characteristics
- STYLE: names for generally-recognized categories of styles ('preppy', 'boho', 'steampunk')

After extracting the fashion-related terms from each of the collection descriptions, I vectorized the output lists of terms using a TFIDF vectorizer. I then took a random sample of 35% of the data (5,000 collections) and ran the sample through a K-means clustering algorithm 50 times, with 50 different random_state values each time for reproducibility. Running the algorithm with the full dataset was very, very slow, so using a smaller subset of the data was necessary for proof of concept. 

I stored the cluster label output for each run, then turned the output into a frequency matrix in which the id numbers of each collection made up both the index and the columns and the values in each cell $x_i,j$ is the number of times out of the 50 runs that the two collections i and j were in the same cluster. This frequency matrix is available for experimentation in the repo under the name freq_5000.csv.

This frequency matrix is then converted into a network adjacency matrix, in which the values in each cell become the weight of the link between the two collection nodes. With a manual threshold in place, collections only form a link if they were in the same cluster more than 50% of the time in order to limit noise. We then apply a Greedy Modularity Optimizer community detection algorithm to the network to identify groups of similar collections. These group assignments become the primary informant to evaluating which collections are the most similar. 

#### Brand longevity analysis
As for the evaluation of brand longevity, I created a Recurrent Neural Network to predict the prevalence score of designers based on their collection descriptions. As the raw prevalence score is a continuous target variable, I utilized the class assignments (based on percentiles) to allow for classification. However, rather than using the class assignments themselves, I opted to turn the problem into a binary classification (1 for high (95th percentile or above) prevalence, 0 for low prevalence). This was due in part to the fact that the prevalence score is not normally distributed, so there are significantly more observations associated with certain levels of prevalence than with others, so even though the classes are based on percentiles the percentiles have very limited differences. 

Were I to have more time or data, I'd be interested in doing a regression on designer metadata in order to predict prevalence. 

In order to better understand which features of a collection contribute the most to designer prevalence, I created a Naive Bayes model and fit it on just the descriptors (fashion related terms extracted from the description), then tracked the model coefficients to see which words specifically had the biggest impact. 


## Results

### Exploratory Data Analysis
The initial exploration of the data provided excellent context for the makeup data itself, as well as several aspects that validate our assumptions. 

![Wordcloud](https://github.com/bellagracefinck/trend_cycle/blob/main/images/wc.png)
Albeit not the most effective visualizations, word clouds help communicate the "main idea" of a dataset and demystify the type of language we are seeing in the archival reviews. Despite the NLP jargon present in this report, the vocabulary of the dataset is not particularly complex or advanced, which is helpful when lemmatizing words. (Note: All words in the visualizations have been lemmatized, or shortened to their root, so words like "create" and "created" are shortened to "creat." This hopefully explains some of the bizarre spellings you see here and in later tables.

![Figure 1](https://github.com/bellagracefinck/trend_cycle/blob/main/images/fig-1.png)
In figure 1, it becomes clear that the majority of our data is coming from the past 15 years. There is an additional dip in 2020 as a result of the COVID-19 pandemic slowing down in-person seasons, but the number of collections appears to be back on the incline post-pandemic. 

![Figure 2](https://github.com/bellagracefinck/trend_cycle/blob/main/images/fig-2.png)
In the top left plot, we notice that there are many designers with very few collections and very few designers with many collections. This indicates to us that the distribution of number of collections follows the power law distribution, and would therefore be suitable for the modelling of a real world network. We can see in the top right plot that there are similarly very few designers who have been around for longer than before the 2010s and many designers who first appeared in the archive in the mid to late 2010s.

In the bottom half of the figure, we compare consistency and prevalence scores. The biggest difference to note between the two is that the inclusion of the prevalence penalty term causes the scale of scores to go from 0 to 1 to -1 to 1. In comparison to the consistency plot, the prevalence scores are more evenly spread out despire there still being a higher number of designers at either end of the spectrum (around -1 to around 1).

![Figure 3](https://github.com/bellagracefinck/trend_cycle/blob/main/images/fig-3.png)
Figure 3 shows the distribution of review lengths, with a slight right skewed but a mostly bell-shaped curve. 

![Figure 4](https://github.com/bellagracefinck/trend_cycle/blob/main/images/fig-4.png)
Figure 4 is an example of the temporal nature of fashion trends. As tweed, a material more commonly associated with formal, often office-related attire, decreases in popularity, silk and lace, materials frequently associated with more romantic or delicate pieces, rise. This is just one example, but the code to recreate this plot with any words of your choosing is available in the EDA notebook. 

### TF-IDF analysis
![Term frequency by years](https://github.com/bellagracefinck/trend_cycle/blob/main/images/term_freq.png)
Each column is made up of the words with the top ten highest normalized tf-idf scores from the time. 

### Classifying collection seasons with Naive Bayes and Support Vector Classifier models
In order to check on the usefulness of the data, I created two supervised learning models to predict the season of different designer collections based on the processed descriptions. The Naive Bayes model was able to achieve a validation accuracy of 0.8311 while the SVC model achieved a validation accuracy of 0.8727. Given the content of the dataset itself, the fact that models were successfully labelling seasons at these accuracy rates is not exactly surprising, but it was helpful for me to validate the usability of the data in a machine learning context.

### Named Entity Recognition
After extracting the fashion entities from each collection description, I decided to plot the proportion of word types over time to see if there were any trends in the way people wrote about fashion. 
![NER](https://github.com/bellagracefinck/trend_cycle/blob/main/images/label_prop.png)

### Applied K-means clustering & network generation

![Frequency matrix](https://github.com/bellagracefinck/trend_cycle/blob/main/images/freq.png)
Here is the frequency/adjacency matrix used to create the network graph. The index and column titles are the id numbers of the collections, which later become the names of the nodes. The value in each cell represents the number of times the nodes at the index and column title were in the same cluster (ex: collections 7120 and 3821 were clustered together 30 times out of 50 runs).

![Network](https://github.com/bellagracefinck/trend_cycle/blob/main/images/graph2.png)
After visualizing the network in Gephi, we can see there are clear groups of designer collections that were clustered together frequently across the runs. Nodes are sized by their weighted degree (the sum of all edge weights connected to that node) and colored in reverse rainbow order by their seasonyear value (1990.0 to 2023.1 goes from blue --> green --> yellow --> orange --> red --> purple). The edge thickness and color both correspond to the weight. 

![Sample 1](https://github.com/bellagracefinck/trend_cycle/blob/main/images/samp1.png)![Sample 2](https://github.com/bellagracefinck/trend_cycle/blob/main/images/samp2.png)
![Sample 3](https://github.com/bellagracefinck/trend_cycle/blob/main/images/samp3.png)![Sample 4](https://github.com/bellagracefinck/trend_cycle/blob/main/images/samp4.png)

Here are four example outputs of the most significant words in different clusters (measured by normalized tfidf). 

### Prediction of designer status with Recurrent Neural Network and Naive Bayes models

#### RNN
We fit an RNN model to the data to predict whether the collection designer is part of the top 95th percentile of designers in terms of prevalence (y = 1 if designer prevalence in 95th-100th percentile, y = 0 otherwise). The (already pre-processed) text is vectorized, embedded, and turned into a TensorFlow dataset. The data is then batched and run through the model (a combination of Dense, LTSM, and Dropout layers). We utilize activation functions that work well with binary classification, like 'sigmoid.'

![RNN History](https://github.com/bellagracefinck/trend_cycle/blob/main/images/rnn_history.png)

As shown above, the validation accuracy hangs around the 0.7335 mark before decreasing slightly, while the training accuracy increases rapidly in the first epoch and then very gradually over time. Though the model becomes slightly overfit by the end, the validation accuracy and training accuracy are within 0.05 of each other, which is a good sign. Additionally, loss is almost entirely minimized for both training and validation sets. 

#### NB
The purpose of the Naive Bayes model in this context is aimed at a slightly different goal but trained on the same data. Rather than trying to predict designer status, the NB model is trained on the fashion-related words extracted by the NER model to try and identify specific collection attributes that contribute to or detract from the long term success of designers. 

I additionally made the decision to remove the GARMENT_TYPE descriptors from the training set because I was less interested in the types of garments than the features, colors, and other characteristics that defined different collections and set them apart.

This model achieves an accuracy score of 0.72 and provides context as to which words contributed the most to identifying the top designers vs. the normal designers. 

![Words NB](https://github.com/bellagracefinck/trend_cycle/blob/main/images/NBwords.png)


## Discussion

### NER 
In the stacked bar chart, there are several visible trends in the makeup of descriptions. Describing collections by their "style," or the niche of fashion that they either fall into or take inspiration from was quite popular in the 1990s but fell out of favor in the early 2000s. We start to see an increase in popularity around 2020, possibly as a result of the internet's influence on fashion. With the rise in trend-related companies such as Pinterest which rely on computer vision techniques in order to improve their recommendation systems, more and more images are being categorized and labelled. With the introduction of these categories, however, though they are not always visible to the consumer, people begin to take note of which keywords produce the output they want the most and create mental links between the output (say, a white lace dress) and the categorical search terms ('romantic'). This kind of "aesthetic" culture has become greater than just style-defining - individuals recognize these niches as not just a formula for what to wear, but an identity. As a result, it comes as no surprise that collection descriptions are beginning to include more style categorizations, as brands seek to capitalize on mankind's innate need to self-categorization.

### Trend cycle network
In the network analysis portion of the study, the K-means output produced a relatively sparse frequency matrix. For my purposes, this was a good thing, as my goal was to uncover groups of commonly clustered collections. Additionally, including the manual filtering of a 50% threshold (edge weight > 25) helped to improve signal in community detection and further separate preexisting clusters from proportionally less relevant/connected nodes. 

After adding node attributes to the network based on designer name, season-year, and class, we are able to visualize the data in Gephi and filter by different attributes. I chose to use the eigenvector centrality to measure the influence of certain nodes/collections and found that the top five most influential collections in the sample of 5000 were Max Mara's Fall 2021 collection, Libertine's Fall 2012 collection, Ellery's Fall 2019 collection, Tanya Taylor's Fall 2017 collection, and Phoebe English's Spring 2020 collection. Out of these designers, only Phoebe English falls within the highest class of prevalence scores. This leads me to wonder if the most prevalent designers are less inclined to join clusters in a clustering algorithm because their prevalence relies on an element of originality. Similarly to how Miranda Priestly remarked, Oscar de la Renta (who is a prevalent designer in this dataset) started the trickle-down of cerulean in the fictional world of *The Devil Wears Prada*, perhaps an application like this best serves to recognize the second step of the trickle-down effect, when multiple designers adopt a feature or trend, rather than identifying the root of the trend itself. 

For curiosity's sake, I performed TF-IDF analysis on the collection descriptions of  four sample communities identified by the greedy modularity community detection algorithm. While not entirely different, there are certain elements of each community that seem to characterize the vibe of collections included in the community itself. For example, in Sample 3, the inclusion of simple, lightweight fabrics such as gauze and linen fall in line with the general style of minimalism and sustainability. Comparatively, Sample 2's mentions of country, western, and lace seem to evoke a similar sense of cohesion. Were this a clothing classification dataset, the juxtaposition of country and punk in this output may have sparked more cause for concern, but given the extremely imaginative nature of designer fashion, it is likely that the terms are closer in nature than we would otherwise assume. 

### Designer prevalence prediction
I initially trained the RNN on the full collection descriptions and attempted to predict whether or not the designer of a collection was part of the top 5% of designers by prevalence score. The model took a long time to fit and often ended up being overfit to the training data regardless of how I tuned the hyperparameters. I  eventually decided to switch the training data to the extracted fashion-specific terms from each collection and was quite pleased with the results. The model was faster, minimized both training and validation loss, and ended with a balanced accuracy score between training and validation sets. 

While the RNN is more complex, it has less interpretability than a simpler model like the Multinomial Naive Bayes model that I trained on the same training dataset. The NB model output includes coefficients that correspond to the words in the dataset and their respective probabilities. Because the coefficients are all calculated by taking the log of some value between 0 and 1 (the probability of class a given the word's presence in the dataset), all coefficients are negative. However they can be interpreted as such: the smaller the absolute value of the coefficient, the higher the probability of being class 1; the larger the absolute value, the lower the probability of being class 1. 

I was interested to see that the most significant informants of being a top designer were all words that are generally recognized as timeless attributes, like leather, silk, and lace. On the other hand, the terms that would most commonly flag a collection as a non-top/normal designer included ultra-specific and experimental features like metallic, velveteen, and tencel. This observation implies that the designers who have seen the most long term success (as measured by prevalence score) are not necessarily the ones who are the most experimental or groundbreaking, but rather the ones who produce clothing that will remain timeless and classic. 


## Conclusion

While my research was somewhat limited in scope by the non-exhaustiveness of my dataset and computational resources available on my laptop, I would be lying if I said I was not pleased with my findings. My network model was able to discover communities of similar collections over time and track the influence of certain collections on those that came after. Through the implementation of my Named Entity Recognition model and creation of NLP classifiers, I was able to discover that the most prevalent designers remain prevalent by refusing to date themselves through participation in flashy trends or attempting to deviate too far from the norm. Rather, they elevate and reimagine the pre-existing standard with high quality craftmanship and materials. 

One of the primary ideas for future research I'd personally like to expand upon is the concept of training a large language model on the dataset and using it for text generation. Using an LLM to generate a prediction for what the next collections from top designers will contain would ideally provide an opportunity to see if the patterns I've recognized in my research lend themselves in any way into the forecasting of trends. It would also allow me to experiment with AI-based trend forecasting. I'd additionally like to continually add to my NER training dataset in order to bolster the robustness of model training and incorporate new trends and clothing features as they are invented. 

I suppose above all else, this project has confirmed to me that *The Devil Wears Prada* was more of a prophecy than a film. Trends will come and go, but influence is forever. 


## References

Bae, H., Cho, S. Y., Yoo, J., & Seok-Bae Yun. (2023). Mathematical modeling of trend cycle: Fad, fashion and classic. [Unpublished manuscript](https://www.proquest.com/working-papers/mathematical-modeling-trend-cycle-fad-fashion/docview/2804142334/se-2?accountid=15053). Retrieved from ProQuest.

Chapter 3: Processing Pipelines. (n.d.). *Advanced NLP with spaCy*. [Advanced NLP with spaCy](https://course.spacy.io/en/chapter3).

Emerson, S. (2022). Gen z’s imitations of life: Aesthetics, self and Pinterest. *University Wire*. [University Wire](https://www.proquest.com/wire-feeds/gen-z-s-imitations-life-aesthetics-self-pinterest/docview/2623033840/se-2?accountid=15053).

Frankel, D. (2006). *The Devil Wears Prada*. Fox 2000 Pictures.

Masse, D. (n.d.). Davidmasse/blog-fashion-system: NLP analysis of fashion trends. GitHub. [GitHub](https://github.com/davidmasse/blog-fashion-system/tree/master).

Shi, M., Chussid, C., Yang, P., Jia, M., Van, D. L., & Cao, W. (2021). The exploration of artificial intelligence application in fashion trend forecasting. *Textile Research Journal, 91*(19-20), 2357-2386. [doi:10.1177/00405175211006212](https://doi.org/10.1177/00405175211006212).

Stanford NLP Group. (n.d.). The Stanford NLP Group. *The Stanford Natural Language Processing Group*. [Stanford NLP Group](https://nlp.stanford.edu/software/CRF-NER.shtml).

Vogue. (n.d.). Fashion shows. *Vogue*. Retrieved from [https://www.vogue.com/fashion-shows](https://www.vogue.com/fashion-shows).

Wolk, C. (2020, March 10). Strike a pose: Reclassifying Vogue’s runway coverage using latent Dirichlet allocation (LDA). *Medium*. [Medium](https://medium.com/@catherinemwolk/strike-a-pose-reclassifying-vogues-runway-coverage-using-latent-dirichlet-allocation-lda-d023a4681687).

Zhao, L., Li, M., & Sun, P. (2024). Neo-fashion: A data-driven fashion trend forecasting system using catwalk analysis. *Clothing and Textiles Research Journal, 42*(1), 19-34. [doi:10.1177/0887302X211004299](https://doi.org/10.1177/0887302X211004299).


## Appendices
This section includes additional information that may be useful to readers, such as detailed descriptions of the data sources, mathematical derivations, or additional statistical analyses.

Here is a subset of the 'collections' dataframe, which contains the text data for each collection.
![collections df](https://github.com/bellagracefinck/trend_cycle/blob/main/images/collections.png)

Here is a subset of the 'designers' dataframe, which contains designer-specific values such as their first season and total number of collections as well as calculated metrics like prevalence and consistency.
![designers df](https://github.com/bellagracefinck/trend_cycle/blob/main/images/designers.png)

Here are summary statistics for the 'designers' dataframe. 
![designers sum](https://github.com/bellagracefinck/trend_cycle/blob/main/images/sumstat_designers.png)