import pandas as pd
from sklearn.cluster import KMeans


def _data():
   #reading the data provided
    articles_df = pd.read_csv('shared_articles.csv')
    print(articles_df.shape)
    
    #Article shared or article removed at a particular timestamp.
    #counting the number of articles shared of removed
    print(articles_df['eventType'].value_counts())

    #filter out the articles removed as they will not help in the recommendation.
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    print(articles_df.shape)
    

    #Title: Title/headline of the articles. 
    print(articles_df['title'].head(5))
    print(articles_df['title'].isnull().sum(axis=0))
    print(articles_df['title'].isna().sum(axis=0))

    #Text: Content of the articles.

    print(articles_df['text'].head(5))
    print(articles_df['text'].isnull().sum(axis=0))
    print(articles_df['text'].isna().sum(axis=0))

    articles_df = pd.DataFrame(articles_df, columns=[ 'content', 'lang', 'title','text'])


def user_data():
    #reading the data from the profile
    profile_df = pd.read_csv('entered_profile.csv')
    print(profile_df.shape)

    #filter out the articles removed as they will not help in the recommendation.
    profile_df = profile_df[profile_df['eventType'] == 'CONTENT SHARED']
    print(profile_df.shape)
    

    #Title: Title/headline of the articles. 
    print(profile_df['locality'].head(5))
    print(profile_df['locality'].isnull().sum(axis=0))
    print(profile_df['locality'].isna().sum(axis=0))
    user_locality=profile_df['locality']







if __name__ == '__main__':
    _data()
