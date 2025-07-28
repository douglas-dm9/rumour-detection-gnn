import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


class Load_Rumours_Dataset_filtering_since_first_post:
    """
    Class for loading, preprocessing, and splitting a rumour detection dataset 
    based on reply posts and their timing, filtering data up to a specified cutoff 
    since the first post.

    Attributes:
        file_path_replies (str): Path to the pickled replies DataFrame.
        file_path_posts (str): Path to the pickled posts DataFrame.
        time_cut (float): Time cutoff in minutes for filtering replies/posts.
        scaler_posts (RobustScaler): Scaler used to normalize numerical post features.
    """

    def __init__(self, file_path_replies, file_path_posts, time_cut):
        """
        Initializes the data loader with file paths and time cutoff.

        Args:
            file_path_replies (str): Path to the replies pickle file.
            file_path_posts (str): Path to the posts pickle file.
            time_cut (float): Time window in minutes since the first post to include replies.
        """
        self.file_path_replies = file_path_replies
        self.file_path_posts = file_path_posts
        self.time_cut = time_cut
        self.scaler_posts = RobustScaler()

    def load_data(self):
        """
        Loads the pickled replies and posts datasets into memory.
        """
        self.df_replies = pd.read_pickle(self.file_path_replies)
        self.df_posts = pd.read_pickle(self.file_path_posts)

    def process_data(self):
        """
        Processes the loaded datasets by:
            - Filtering replies within the specified time cutoff.
            - Computing features such as number of replies and time since post.
            - One-hot encoding 'verified' column.
            - Splitting the data into train, validation, and test sets.
            - Scaling numerical features.
            - Returning processed and scaled train and test DataFrames.
        """
        features = ['followers', 'favorite_count', 'retweet_count', 'verified', 'rumour', 'id', 'embeddings_avg']

        # Compute time since first post for replies
        self.df_replies['min_since_fst_post'] = round(
            (self.df_replies['time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)
        self.df_replies['reply_min_since_fst_post'] = round(
            (self.df_replies['reply_time'] - self.df_replies['time'].min()).dt.total_seconds() / 60, 2)

        # Aggregate replies by post
        grouped_replies = self.df_replies.groupby(['id', 'min_since_fst_post']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge with post features
        self.df_posts = self.df_posts[features].merge(grouped_replies, on="id", how="inner").fillna(0)

        # One-hot encode verification status
        self.df_posts['verified'] = self.df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        self.df_posts = pd.concat([self.df_posts, pd.get_dummies(self.df_posts["verified"], dtype=int)], axis=1)
        self.df_posts.drop(["verified"], axis=1, inplace=True)
        self.df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        # Train/validation/test split
        train, not_train = train_test_split(self.df_posts, test_size=0.3, random_state=42, stratify=self.df_posts['rumour'])
        val, test = train_test_split(not_train, test_size=0.5, random_state=42, stratify=not_train['rumour'])

        # Scale selected numeric features for training
        train_features = train[["followers", "favorite_count", "retweet_count", "no_verified", "verified", 
                                "rumour", "embeddings_avg", "replies", "first_time_diff", "min_since_fst_post"]]
        scaled_numeric = self.scaler_posts.fit_transform(train_features[['followers', 'favorite_count',
                                                                          'retweet_count', 'first_time_diff', 'replies']])
        scaled_data = pd.DataFrame(scaled_numeric, columns=['followers', 'favorite_count', 'retweet_count', 'first_time_diff', 'replies'])
        scaled_data['no_verified'] = train['no_verified'].values
        scaled_data['verified'] = train['verified'].values
        scaled_data['embeddings_avg'] = train['embeddings_avg'].values
        scaled_data['rumour'] = train['rumour'].values
        scaled_data['min_since_fst_post'] = train['min_since_fst_post'].values

        self.train_dataset = scaled_data

        # Reload and filter posts/replies for test/val based on time_cut
        test_val_df_replies = pd.read_pickle(self.file_path_replies)
        test_val_df_posts = pd.read_pickle(self.file_path_posts)

        test_val_df_posts = test_val_df_posts[~test_val_df_posts.id.isin(train.id)]
        test_val_df_replies = test_val_df_replies[~test_val_df_replies.id.isin(train.id)]

        test_val_df_replies['min_since_fst_post'] = round(
            (test_val_df_replies['time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)
        test_val_df_replies['reply_min_since_fst_post'] = round(
            (test_val_df_replies['reply_time'] - test_val_df_replies['time'].min()).dt.total_seconds() / 60, 2)

        # Filter by time_cut
        test_val_df_replies = test_val_df_replies[
            (test_val_df_replies.reply_min_since_fst_post <= self.time_cut) &
            (test_val_df_replies.min_since_fst_post <= self.time_cut)]

        # Group replies again
        grouped_replies = test_val_df_replies.groupby(['id', 'min_since_fst_post']).agg(
            replies=('time_diff', 'count'),
            first_time_diff=('time_diff', 'first')
        ).reset_index()

        # Merge with posts and apply same transformations
        test_val_df_posts = test_val_df_posts[features].merge(grouped_replies, on="id", how="inner").fillna(0)

        test_val_df_posts['verified'] = test_val_df_posts['verified'].astype(str).replace({'True': '1', 'False': '0'}).astype(int)
        test_val_df_posts = pd.concat([test_val_df_posts, pd.get_dummies(test_val_df_posts["verified"], dtype=int)], axis=1)
        test_val_df_posts.drop(["verified"], axis=1, inplace=True)
        test_val_df_posts.rename(columns={1: 'verified', 0: 'no_verified'}, inplace=True)

        # Reindex based on existing validation and training IDs
        test_val_df_posts = test_val_df_posts.merge(pd.concat([val, train])[['id']].reset_index(), on='id', how='left')
        test_val_df_posts.set_index('index', drop=True, inplace=True)

        post_features = test_val_df_posts[["followers", "favorite_count", "retweet_count", "no_verified", "verified",
                                           'replies', "first_time_diff", "embeddings_avg", "rumour", "min_since_fst_post"]]

        scaled_numeric = self.scaler_posts.transform(post_features[['followers', 'favorite_count',
                                                                    'retweet_count', 'first_time_diff', 'replies']])
        scaled_data = pd.DataFrame(scaled_numeric, columns=['followers', 'favorite_count', 'retweet_count',
                                                            'first_time_diff', 'replies'])
        scaled_data['no_verified'] = post_features['no_verified'].values
        scaled_data['verified'] = post_features['verified'].values
        scaled_data['embeddings_avg'] = post_features['embeddings_avg'].values
        scaled_data['rumour'] = post_features['rumour'].values
        scaled_data['min_since_fst_post'] = post_features['min_since_fst_post'].values

        self.test_dataset = scaled_data

    def get_final_dataframes(self):
        """
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The processed training and test datasets.
        """
        return self.train_dataset, self.test_dataset
