import newspaper
import transformers
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns


# Crawl the CNN website using the newspaper web crawling package.
link = 'https://www.cnn.com'
# Scans the webpage and finds all the links on it.
page_features = newspaper.build(link, language='en', memoize_articles=False)
# Initialize a list for article titles and text.
title_text = list()
# The page_features object contains article objects that are initialized with links to the web pages.
for article in page_features.articles:
    try:
        # Each article must be downloaded, then parsed individually.
        # This loads the text and title from the webpage to the object.
        article.download()
        article.parse()
        # Keep the text, title and URL from the article and append to a list.
        title_text.append({'title':article.title,
                           'body':article.text,
                           'url': article.url})
    except:
        # If, for any reason the download fails, continue the loop.
        print("Article Download Failed.")

# Save as a dataframe to avoid excessive calls on the web page.
articles_df = pd.DataFrame.from_dict(title_text)
articles_df.to_csv(r'CNN_Articles_Oct15_21.csv')

# Load the dataframe from the checkpoint.
articles_df = pd.read_csv(r'CNN_Articles_Oct15_21.csv')

# Drop any NaNs from the dataframe.
articles_df = articles_df.dropna().iloc[:,1:]

# Plot the distribution of body and title length.
fig, (ax1,ax2) = plt.subplots(1,2)
articles_df['title'].apply(lambda x: len(x)).hist(ax=ax1)
ax1.set_title('Title Length Distribution')
articles_df['body'].apply(lambda x: len(x)).hist(ax = ax2)
ax2.set_title('Body Length Distribution')

# Get the character length of each article.
len_df = articles_df.applymap(lambda x:len(x))
# Drop articles where the title is longer than the body (for example, video articles).
len_df['title_gt_body'] = len_df['title'] > len_df['body']
print(len_df['title_gt_body'].sum()/len_df.shape[0])
# Drop non-english articles that were downloaded.
len_df['spanish'] = articles_df['url'].astype(str).str.contains('cnnespanol|arabic')
print(len_df['spanish'].sum()/len_df.shape[0])
len_df['mask'] = len_df['title_gt_body']|len_df['spanish']
print(len_df['mask'].sum()/len_df.shape[0])

# Plot histograms of the article body and title lengths after cleaning.
fig, (ax1,ax2) = plt.subplots(1,2)
len_df[~len_df['mask']]['title'].hist(ax= ax1)
len_df[~len_df['mask']]['body'].hist(ax= ax2)

# Finish the cleaning, remove bad samples.
article_df_clean = articles_df[~len_df['mask']]
len_clean = len_df[~len_df['mask']]

# Setup a pipeline with a Pytorch/HuggingFace NLP summarization model.
# This CNN can summarize up to 1024 words.
smr_bart = transformers.pipeline(task="summarization", model="facebook/bart-large-cnn")

# Initialize a list for the summaries.
summary_list = list()
# Get a summary for each article.
for ind,x in article_df_clean.iterrows():
    # print(len(x['title']))
    # Split the text into words.
    body = x['body']
    body = body.split()
    try:
        # Check the word count, only analyze up to the first 750 words.
        if len(body)>750:
            body = body[0:750]#:1023]
        # Put the words back together into one article for the NLP pipeline.
        body = ' '.join(body)
        # Calculate the NLP summary using the model pipeline.
        # Make the summary as long as the title.
        summary = smr_bart(body,max_length=len(x['title']))[0]['summary_text']
        summary_list.append({'index':ind,'summary':summary})
    except:
        # If there are any failures, print them for debugging later.
        print('Failure on Index# '+str(ind))

# Make the summaries into a dataframe.
summary_df = pd.DataFrame.from_dict(summary_list).set_index(keys='index')

# Merge the summaries back with the articles.
article_summary_df = pd.merge(summary_df,article_df_clean,
                              left_index=True,right_index=True,how='left')
article_summary_df.to_pickle(r'CNN_Articles_wSummaries_Oct15_21.pkl')

article_summary_df = pd.read_pickle(r'CNN_Articles_wSummaries_Oct15_21.pkl')
# Use textblob package to estimate the polarity of the title, body and summary.
polarity_df = article_summary_df[['title','body','summary']].\
    applymap(lambda x: TextBlob(x).sentiment.polarity)
polarity_df.hist()

print(polarity_df.describe())

# Plot
g = sns.PairGrid(polarity_df)
g.map_upper(sns.regplot)
g.map_lower(sns.regplot)
g.map_diag(sns.histplot, kde=True)

# Pearson correlation coefficient (little r) squared suggests fit quality.
print(polarity_df.corr().pow(2))