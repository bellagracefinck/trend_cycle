## The Devil Wears Prada, I Wear Whatever's on Sale: NLP Applications in Fashion Trend Forecasting and Brand Longevity Analysis
### Bella Grace Finck, DATA 340

## Abstract
Abstract: This section provides a brief summary of the project, highlighting the main research question, methodology, results, and conclusions. It should be concise and clear, usually limited to 250-300 words.

The field of fashion trend forecasting has existed for decades, but recent strides in natural language processing and AI have opened countless doors for opportunities to fortify the primarily qualitative work done by forecasting agencies and consumers alike. Through a thorough analysis of Vogue Runway's archive of Ready-to-Wear collections from the past 34 years, this project aims to answer one question: 

>How successful are natural language processing efforts at recognizing the trend cycle in action, identifying the characteristics of fashion designers who achieve long-term success, and using the two to predict trends and deisgner longevity alike?



## Introduction
One of my favorite movie monologues of all time is by Meryl Streep (as fictional fashion magazine editor Miranda Priestly) in the 2006 cult classic The Devil Wears Prada. After a poorly timed joke by her extremely ill-equipped assistant, Streep famously exposes the trend cycle as we know it today while simultaneously decimating Anne Hathaway and shining a light on the true power of the fashion industry leaders in the process. (https://www.youtube.com/watch?v=vL-KQij0I8I)
       
> “You… go to your closet, and you select… I don’t know, that lumpy blue sweater, for instance, because you’re trying to tell the world that you take yourself too seriously to care about what you put on your back, but what you don’t know is that that sweater is not just blue, it’s not turquoise, it’s not lapis, it’s actually cerulean. You’re also blithely unaware of the fact that, in 2002, Oscar de la Renta did a collection of cerulean gowns, and then I think it was Yves Saint Laurent, wasn’t it?… who showed cerulean military jackets. […] And then cerulean quickly showed up in the collections of eight different designers. Then it filtered down through the department stores and then trickled on down into some tragic casual corner where you, no doubt, fished it out of some clearance bin. However, that blue represents millions of dollars of countless jobs, and it’s sort of comical how you think that you’ve made a choice that exempts you from the fashion industry when, in fact, you’re wearing a sweater that was selected for you by the people in this room… from a pile of ‘stuff.’” - *The Devil Wears Prada*, 2006
        
Even as I read the words for the hundredth time, I can feel the prickling sensation I got the first time I ever watched the film. In a world in which the illusion of choice is everywhere, fashion is often advertised as an escape - an opportunity for one to make choices based exclusively on their own intended self-expression. However, just like most things in our society, the vast majority of decisions we make have already been made for us by a group of highly influential elites. 

I have always been interested in fashion trend forecasting and occasionally attempt to qualitatively analyze things myself, but this project is by far my most ambitious endeavor into the world of trend forecasting yet. As I began to research current methodologies in trend forecasting agencies, I discovered that most projects are aimed at evaluating the lifetimes of trends rather than evaluating the characteristics that enable the most successful members of the fashion industry to maintain their success and influence. This makes sense in the context of our efficiency and profit-driven world, but my instincts tell me that trend forecasting as a field would benefit from the greater contextualization of analyzing brand longevity. 

I hope that by discovering similarities between different designer collections and overall seasons I will be able to identify and anticipate potential upcoming trends and that by analyzing designer prevalence over time I will be able to predict the future success of specific designers. 


## Literature review: 
This section provides a comprehensive review of the relevant literature on the topic being studied. It highlights the strengths and weaknesses of previous research and identifies gaps in the current understanding of the topic. [NOTE: Not required but will make an impression. 1-2 paragraphs]
- Trend forecasting
- Trend cycle explained
- AI for trend forecasting
- This work provides additional contextualization 

## Methodology/Dataset

#### Data collection
Data was scraped from the Vogue Runway archive using BeautifulSoup (see data_generation notebook). Although it originated as a print-only publication, Vogue has gradually shifted to include their articles and images in a digital format on their website. They have also begun to digitize collections from before the establishment of their digital presence, dating back to Fall 1988. My dataset is made up of every ready-to-wear collection in the archive since Spring 1990 and through Fall 2023. I have saved the raw article text for each of these collections along with the respective season, year, and designer in my complete.csv file.  

It is important to note that while this dataset is large, it is not exhaustive - particularly in the earlier seasons - due to the fact that Vogue is currently expanding the Vogue Runway archive by digitizing collections originally shot on film and reviews originally written for print.   

Given the aims of my project, I opted to use Ready to Wear collections rather than Couture. This was based on the assumption that RTW collections are truer to the products that designers are actually selling to a consumer base, therefore they are the collections that will influence the trend cycle the most. While all designer clothing is an art form, there tends to be more repetition and influence in RTW collections than in couture collections, in which there is very little practicality required of the clothing itself. 

#### Data format
Collection descriptions are located in the *collections* dataframe, which consists of the columns 'season', 'year', 'seasonyear', 'designer', 'text', 'id', and 'preprocessed_sentences' and is eventually mapped to include 'consistency', 'prevalence', and 'class'.

Designer-specific information is contained in the *designers* dataframe, which consists of 'designer', 'collections' (total number of designer's collections), and 'first_season', 'consistency', 'prevalence', and 'class'. 

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

The *class* variable is a class assignment by percentile of prevalence score. Classes range from 0 to 5 and the 0-20, 20-40th, 40-60th, 60-80th, 80-90th, and 90-100th percentiles, respectively. 

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

This frequency matrix is then converted into a network adjacency matrix, in which the values in each cell become the weight of the link between the two collection nodes. With a manual threshold in place, collections only form a link if they were in the same cluster more than 30% of the time in order to limit noise. We then apply a Louvain community detection algorithm to the network to identify groups of similar collections. These group assignments become the primary informant to evaluating which collections are the most similar. 

#### Brand longevity analysis
As for the evaluation of brand longevity, I created a Recurrent Neural Network to predict the prevalence score of designers based on their collection descriptions. As the raw prevalence score is a continuous target variable, I utilized the class assignments (based on percentiles) to allow for classification. However, rather than using the class assignments themselves, I opted to turn the problem into a binary classification (1 for high (95th percentile or above) prevalence, 0 for low prevalence). This was due in part to the fact that the prevalence score is not normally distributed, so there are significantly more observations associated with certain levels of prevalence than with others, so even though the classes are based on percentiles the percentiles have very limited differences. 

Were I to have more time or data, I'd be interested in doing a regression on designer metadata in order to predict prevalence. 

In order to better understand which features of a collection contribute the most to designer prevalence, I created a Naive Bayes model and fit it on just the descriptors (fashion related terms extracted from the description), then tracked the model coefficients to see which words specifically had the biggest impact. 


## Results
This section presents the findings of the research, including descriptive statistics, tables, and graphs. It should provide a clear and concise summary of the main results, highlighting any patterns or trends observed. [NOTE: 2-4 paragraphs]

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


### Classifying collection seasons with Naive Bayes and Support Vector Classifier models
In order to check on the usefulness of the data, I created two supervised learning models to predict the season of different designer collections based on the processed descriptions. The Naive Bayes model was able to achieve a validation accuracy of 0.8311 while the SVC model achieved a validation accuracy of 0.8727. Given the content of the dataset itself, the fact that models were successfully labelling seasons at these accuracy rates is not exactly surprising, but it was helpful for me to validate the usability of the data in a machine learning context.

###


## Discussion
The discussion section interprets the results of the study in light of the research question and literature review. It should explain how the findings relate to previous research and provide a critical analysis of their implications. [NOTE: 6-10 paragraphs]



## Conclusion
This section summarizes the main findings of the study, restates the research question, and discusses the implications of the research for future research and practice. [NOTE: 1-2 paragraphs]



## References
This section provides a list of all the sources cited in the paper, following a specific citation style (e.g., APA, MLA).

## Appendices
This section includes additional information that may be useful to readers, such as detailed descriptions of the data sources, mathematical derivations, or additional statistical analyses.



