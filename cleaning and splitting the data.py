import pandas as pd
from sklearn.model_selection import train_test_split

import preprocessing


def split_data_into_train_test(df, columns_list):
    X = df[columns_list]
    y = df[['genre']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=123)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    music_df = pd.read_csv("Final data/lyrics_final.csv")
    music_df['cleaned_lyrics'] = music_df.apply(preprocessing.preprocessing, axis=1)
    music_df.to_csv("Final data/cleaned_lyrics.csv")

    # removing NA and other rows
    music_df = preprocessing.data_cleaning(music_df)

    # train-test split
    columns = ['song', 'artist', 'lyrics', 'cleaned_lyrics']
    X_train, X_test, y_train, y_test = split_data_into_train_test(music_df, columns)
    train_df = X_train.join(y_train)
    test_df = X_test.join(y_test)

    train_df.to_csv("Final data/train.csv")
    test_df.to_csv("Final data/test.csv")
