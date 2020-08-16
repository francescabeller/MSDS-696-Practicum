# MSDS-696-Practicum
GitHub repository for MSDS-696 Data Science Practicum

#### References
https://spotipy.readthedocs.io/en/2.13.0/

https://github.com/plamere/spotipy/tree/master/examples

# Spotify Recommendation Algorithm

### Spotify Authentication

```python
# Import libraries
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn as skl
```

After importing the libraries above, we can use `spotipy` to connect to my personal Spotify account (Note: secret key for account is X'd out for privacy).

```python
# Set account info
cid = '81fee852cceb4259910e7d2ff78493c3'
secret = 'XXXXX'
username = 'francescab13'

# Connect and create Spotify instance
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
```

## Data Gathering

#### Retrieve track ID's from 'Like' and 'Dislike' playlists

Next, we will have to retrieve the track IDs from the "Like" and "Dislike" playlists I have created.

```python
# Get data from 'Likes' playlist
good_ids = []
pl_id = 'spotify:playlist:2O6XH1ip37KOllmc1KoYEs'
offset = 0

while True:
    response = sp.playlist_tracks(pl_id,
                                  offset=offset,
                                  fields='items.track.id,total')
    good_ids.append(response['items'])
    offset = offset + len(response['items'])

    if len(response['items']) == 0:
        break

# Flatten list of lists of JSON
good_flatten = []
for sublist in good_ids:
    for item in sublist:
        good_flatten.append(item)

# Check good track ID list
good_flatten[0:5]
```

Checking the `good_flatten` list, we see the following list of dictionaries:
```python
[{'track': {'id': '75Q69chmd8CEZbVsA4CDMm'}},
 {'track': {'id': '38kjIfRtXsUxXyzhsKwX7i'}},
 {'track': {'id': '1YT8xkroYGNLGR4qhuWLC4'}},
 {'track': {'id': '76gYk9g0bZj47NyIKzjLF6'}},
 {'track': {'id': '7tvuLLroI0n6uYBWuFig5d'}}]
 ```
 
 We will do the same for the "Dislike" playlist.
 
 ```python
 # Get data from 'Dislikes' playlist
bad_ids = []
pl_id = 'spotify:playlist:58KlzYsGNQoujtrQc2CU5d'
offset = 0

while True:
    response = sp.playlist_tracks(pl_id,
                                  offset=offset,
                                  fields='items.track.id,total')
    bad_ids.append(response['items'])
    offset = offset + len(response['items'])

    if len(response['items']) == 0:
        break

# Flatten list of lists of JSON
bad_flatten = []
for sublist in bad_ids:
    for item in sublist:
        bad_flatten.append(item)
        
# Check bad track ID list
bad_flatten[0:5]
```

```python
[{'track': {'id': '1YwNlWLf8auhazSQUDQLFU'}},
 {'track': {'id': '1xShPgQbOUa98avWJQFDBY'}},
 {'track': {'id': '3GREm6zSHwKZsJxl0hqbAQ'}},
 {'track': {'id': '0C6EIiQu8CS4eYtOCMEiAd'}},
 {'track': {'id': '0puf9yIluy9W0vpMEUoAnN'}}]
```

#### Get track characteristic data

Now, we will have to retrieve the audio features for each of the tracks.

```python
# Compile list of 'good' track IDs
good_id_list = []
for i in range(0, len(good_flatten)):
    good_id_list.append(good_flatten[i]['track']['id'])
good_id_list = [x for x in good_id_list if x]

# Retrieve track characteristics
good_features = []
for i in range(0, len(good_id_list)):
    if not good_id_list[i]:
        continue
    else:
        good_features.append(sp.audio_features(good_id_list[i]))

# Flatten JSON list
good_features_flat = []
for sublist in good_features:
    for item in sublist:
        good_features_flat.append(item)
        
# Check 'good' features list
good_features_flat[0:3]
```

We can see a preview of our list of "good" features...

```python
[{'danceability': 0.833,
  'energy': 0.545,
  'key': 2,
  'loudness': -4.004,
  'mode': 0,
  'speechiness': 0.462,
  'acousticness': 0.352,
  'instrumentalness': 0,
  'liveness': 0.0915,
  'valence': 0.541,
  'tempo': 77.035,
  'type': 'audio_features',
  'id': '75Q69chmd8CEZbVsA4CDMm',
  'uri': 'spotify:track:75Q69chmd8CEZbVsA4CDMm',
  'track_href': 'https://api.spotify.com/v1/tracks/75Q69chmd8CEZbVsA4CDMm',
  'analysis_url': 'https://api.spotify.com/v1/audio-analysis/75Q69chmd8CEZbVsA4CDMm',
  'duration_ms': 148494,
  'time_signature': 4},
 {'danceability': 0.596,
  'energy': 0.248,
  'key': 0,
  'loudness': -14.576,
  'mode': 0,
  'speechiness': 0.274,
  'acousticness': 0.794,
  'instrumentalness': 0,
  'liveness': 0.0872,
  'valence': 0.189,
  'tempo': 112.043,
  'type': 'audio_features',
  'id': '38kjIfRtXsUxXyzhsKwX7i',
  'uri': 'spotify:track:38kjIfRtXsUxXyzhsKwX7i',
  'track_href': 'https://api.spotify.com/v1/tracks/38kjIfRtXsUxXyzhsKwX7i',
  'analysis_url': 'https://api.spotify.com/v1/audio-analysis/38kjIfRtXsUxXyzhsKwX7i',
  'duration_ms': 383025,
  'time_signature': 4},
 {'danceability': 0.546,
  'energy': 0.771,
  'key': 3,
  'loudness': -7.56,
  'mode': 0,
  'speechiness': 0.116,
  'acousticness': 0.0117,
  'instrumentalness': 0.351,
  'liveness': 0.335,
  'valence': 0.701,
  'tempo': 130.224,
  'type': 'audio_features',
  'id': '1YT8xkroYGNLGR4qhuWLC4',
  'uri': 'spotify:track:1YT8xkroYGNLGR4qhuWLC4',
  'track_href': 'https://api.spotify.com/v1/tracks/1YT8xkroYGNLGR4qhuWLC4',
  'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1YT8xkroYGNLGR4qhuWLC4',
  'duration_ms': 265947,
  'time_signature': 4}]
  ```
  
  ...and we will do the same for our "bad" features.
  
  ```python
  # Compile list of 'bad' track IDs
bad_id_list = []
for i in range(0, len(bad_flatten)):
    bad_id_list.append(bad_flatten[i]['track']['id'])
bad_id_list = [x for x in bad_id_list if x]

# Retrieve track characteristics
bad_features = []
for i in range(0, len(bad_id_list)):
    if not bad_id_list[i]:
        continue
    else:
        bad_features.append(sp.audio_features(bad_id_list[i]))

# Flatten JSON list
bad_features_flat = []
for sublist in bad_features:
    for item in sublist:
        bad_features_flat.append(item)
        
# Check 'bad' features list
bad_features_flat[0:3]
```
```python
[{'danceability': 0.641,
  'energy': 0.812,
  'key': 0,
  'loudness': -7.945,
  'mode': 0,
  'speechiness': 0.0293,
  'acousticness': 0.155,
  'instrumentalness': 4.43e-05,
  'liveness': 0.078,
  'valence': 0.822,
  'tempo': 112.777,
  'type': 'audio_features',
  'id': '1YwNlWLf8auhazSQUDQLFU',
  'uri': 'spotify:track:1YwNlWLf8auhazSQUDQLFU',
  'track_href': 'https://api.spotify.com/v1/tracks/1YwNlWLf8auhazSQUDQLFU',
  'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1YwNlWLf8auhazSQUDQLFU',
  'duration_ms': 257333,
  'time_signature': 4},
 {'danceability': 0.637,
  'energy': 0.682,
  'key': 11,
  'loudness': -11.625,
  'mode': 1,
  'speechiness': 0.0366,
  'acousticness': 0.0112,
  'instrumentalness': 0.0234,
  'liveness': 0.0473,
  'valence': 0.714,
  'tempo': 129.983,
  'type': 'audio_features',
  'id': '1xShPgQbOUa98avWJQFDBY',
  'uri': 'spotify:track:1xShPgQbOUa98avWJQFDBY',
  'track_href': 'https://api.spotify.com/v1/tracks/1xShPgQbOUa98avWJQFDBY',
  'analysis_url': 'https://api.spotify.com/v1/audio-analysis/1xShPgQbOUa98avWJQFDBY',
  'duration_ms': 224907,
  'time_signature': 4},
 {'danceability': 0.679,
  'energy': 0.829,
  'key': 9,
  'loudness': -7.288,
  'mode': 0,
  'speechiness': 0.0604,
  'acousticness': 0.0875,
  'instrumentalness': 2.46e-06,
  'liveness': 0.318,
  'valence': 0.812,
  'tempo': 119.96,
  'type': 'audio_features',
  'id': '3GREm6zSHwKZsJxl0hqbAQ',
  'uri': 'spotify:track:3GREm6zSHwKZsJxl0hqbAQ',
  'track_href': 'https://api.spotify.com/v1/tracks/3GREm6zSHwKZsJxl0hqbAQ',
  'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3GREm6zSHwKZsJxl0hqbAQ',
  'duration_ms': 232080,
  'time_signature': 4}]
  ```
  
  #### Create dataframes for 'liked' and 'disliked' tracks with audio features
  
  From here, we will create separate dataframes for our "like" and "dislike" playlists.
  
  ```python
  # Create 'Like' dataframe
like_df = pd.DataFrame.from_records(good_features_flat)

# Retrieve song and artist names to add to dataframe
good_song_names = []
good_artists = []
for index, row in like_df.iterrows():
    try:
        response = sp.track(str(row['uri']))
        good_song_names.append(response['name'])
        good_artists.append(response['artists'][0]['name'])
    except SpotifyException as e:
        good_song_names.append('Unknown')
        good_artists.append('Unknown')

# Create 'song_name' and 'artist' columns
like_df['song_name'] = good_song_names
like_df['artist'] = good_artists
```
```python
# Create 'Dislike' dataframe
dislike_df = pd.DataFrame.from_records(bad_features_flat)

# Retrieve song and artist names to add to dataframe
bad_song_names = []
bad_artists = []
for index, row in dislike_df.iterrows():
    try:
        response = sp.track(str(row['uri']))
        bad_song_names.append(response['name'])
        bad_artists.append(response['artists'][0]['name'])
    except SpotifyException as e:
        bad_song_names.append('Unknown')
        bad_artists.append('Unknown')

# Create 'song_name' and 'artist' columns
dislike_df['song_name'] = bad_song_names
dislike_df['artist'] = bad_artists
```

## Exploratory Data Analysis

Now that we have created our two dataframes, we can perform exploratory data analysis. There is an amazing Python package called `pandas_profiling` which creates an interactive HTML report for a `pandas` dataframe. To view these HTML reports for the "like" and "dislike" dataframes, take a look at the Jupyter Notebook in this repository.

```python
#Importing the function
from pandas_profiling import ProfileReport
```
```python
like_profile = ProfileReport(like_df, title='Liked Songs Pandas Profiling Report', explorative = True)
dislike_profile = ProfileReport(dislike_df, title='Disliked Songs Pandas Profiling Report', explorative = True)
```

## Data Visualization

Now, we will want to compare our two datasets using some visualizations to see if there are any notable differences between the two. We'll start by creating two lists to hold our feature columns.


```python
# Create list of audio feature column names
trait_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
             'liveness', 'valence']
discrete_trait_cols = ['key', 'mode', 'tempo', 'time_signature']
```
Now we can move on to plotting.

#### Dist Plots

##### Tempo

```python3
sns.distplot(like_df['tempo'], color='indianred', axlabel='Tempo')
sns.distplot(dislike_df['tempo'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_tempo.png?raw=true)

##### Danceability

```python3
sns.distplot(like_df['danceability'], color='indianred', axlabel='Danceability')
sns.distplot(dislike_df['danceability'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_danceability.png?raw=true)

##### Energy

```python3
sns.distplot(like_df['energy'], color='indianred', axlabel='Energy')
sns.distplot(dislike_df['energy'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_energy.png?raw=true)

##### Loudness

```python3
sns.distplot(like_df['loudness'], color='indianred', axlabel='Loudness')
sns.distplot(dislike_df['loudness'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_loudness.png?raw=true)

##### Speechiness

```python3
sns.distplot(like_df['speechiness'], color='indianred', axlabel='Speechiness')
sns.distplot(dislike_df['speechiness'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_speechiness.png?raw=true)

##### Acousticness

```python3
sns.distplot(like_df['acousticness'], color='indianred', axlabel='Acousticness')
sns.distplot(dislike_df['acousticness'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_acousticness.png?raw=true)

##### Instrumentalness

```python3
sns.distplot(like_df['instrumentalness'], color='indianred', axlabel='Instrumentalness')
sns.distplot(dislike_df['instrumentalness'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_instrumentalness.png?raw=true)


##### Liveness

```python3
sns.distplot(like_df['liveness'], color='indianred', axlabel='Liveness')
sns.distplot(dislike_df['liveness'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_liveness.png?raw=true)

##### Valence

```python3
sns.distplot(like_df['valence'], color='indianred', axlabel='Valence')
sns.distplot(dislike_df['valence'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_valence.png?raw=true)

##### Key

```python3
sns.distplot(like_df['key'], color='indianred', axlabel='Tempo')
sns.distplot(dislike_df['key'], color='mediumslateblue')
plt.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/distplots_key.png?raw=true)

#### Frequency Plots

##### Time Signature & Mode

```python3
fig, ax = plt.subplots(2, 2)
sns.countplot(like_df['time_signature'], ax=ax[0,0])
sns.countplot(dislike_df['time_signature'], ax=ax[0,1])
sns.countplot(like_df['mode'], ax=ax[1,0])
sns.countplot(dislike_df['mode'], ax=ax[1,1])
fig.show()
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/freq_plots_mode_ts.png?raw=true)

#### Pair Plots

##### 'Like' Playlist

```python3
like_pairplot = sns.pairplot(like_df[trait_cols])
```

![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/like_pairplot.png?raw=true)


##### 'Dislike' Playlist

```python3
dislike_pairplot = sns.pairplot(dislike_df[trait_cols])
```

![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/dislike_pairplot.png?raw=true)

#### Correlation Heatmaps

##### 'Like' Playlist
```python3
# Calculate correlations
corr = like_df[trait_cols].corr()
 
# Heatmap
like_corr_heatmap = sns.heatmap(corr)
figure = like_corr_heatmap.get_figure() 
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/like_corr_heatmap.png?raw=true)

##### 'Dislike' Playlist
```python3
# Calculate correlations
corr = dislike_df[trait_cols].corr()
 
# Heatmap
dislike_corr_heatmap = sns.heatmap(corr)
figure = dislike_corr_heatmap.get_figure() 
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/dislike_corr_heatmap.png?raw=true)


## Model Creation/Training

#### Preparation

First, before we can start training models, we will need to add the like/dislike boolean tags to the data and generate a combined dataframe.

```python3
# Assign tags to liked and disliked songs
like_df['target'] = 1
dislike_df['target'] = 0

# Create combined dataframe
dfs = [like_df, dislike_df]
full_df = pd.concat(dfs)
```

Next, we will use the `train_test_split` function from `sklearn` to split our full dataset into training and test sets.

```python3
# Creating training/test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(full_df, test_size = 0.15)
```

The last bit of preparation will be to define our features and targets to finish creating our training/test sets.

```python3
#Define feature sets
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
              'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
x_train = train[features]
y_train = train["target"]
x_test = test[features]
y_test = test["target"]
```

#### Decision Tree

The first model we will start with will be a decision tree classifier.

```python3
dtc = DecisionTreeClassifier(criterion='gini', 
                             min_samples_split=100, 
                             max_depth=11)

dt = dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
score = accuracy_score(y_test, y_pred) * 100
print("Accuracy using Decision Tree: ", round(score, 1), "%")
```
This initial model gives us an accuracy of...
```python3
Accuracy using Decision Tree:  58.1 %
```

#### K-Nearest Neighbors

Next, we'll try a k-nearest neighbors model.

```python3
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(3)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
score = accuracy_score(y_test, knn_pred) * 100
print("Accuracy using KNN Tree: ", round(score, 1), "%")
```
Which yielded and accuracy of...
```python3
Accuracy using Knn Tree:  49.3 %
```
This is performing far below the decision tree, so we will not be moving forward with this model.

#### AdaBoost/Gradient Boosting

Now, we will try both an AdaBoost classifier and Gradient Boosting classifier.

```python3
# Import packages
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
```
```python3
ada = AdaBoostClassifier(n_estimators=100)
ada.fit(x_train, y_train)
ada_pred = ada.predict(x_test)
score = accuracy_score(y_test, ada_pred) * 100
print("Accuracy using ada: ", round(score, 1), "%")
```
Our AdaBoost classifier got an accuracy score of...
```python3
Accuracy using ada:  60.5 %
```
This puts it ahead of the original decision tree model, so as of this point, AdaBoost will be the one moving forward.

Meanwhile, our Gradient Boosting classifier did not perform as well.
```python3
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=1, random_state=0)
gbc.fit(x_train, y_train)
predicted = gbc.predict(x_test)
score = accuracy_score(y_test, predicted)*100
print("Accuracy using Gbc: ", round(score, 1), "%")
```
```python3
Accuracy using Gbc:  57.7 %
```

#### Naive Bayes

The last initial model we will try is the Gaussian Naive Bayes.

```python3
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()  # instantiate model
y_pred = gnb.fit(x_train, y_train).predict(x_test)

# get classification metrics
gnb_pred = ada.predict(x_test)
score = accuracy_score(y_test, gnb_pred) * 100
print("Accuracy using GNB: ", round(score, 1), "%")
```
```python3
Accuracy using GNB:  60.1 %
```
We can see that this accuracy score is just slightly below the top-performing AdaBoost.

After looking at all of these initial models, AdaBoost performed the best, so we will move forward with this classifier for further tuning and re-testing.


## Model Tuning

Now, we can tune our AdaBoost model. First, we will import `GridSearchCV` and `KFold` from `sklearn`.
```python3
from sklearn.model_selection import GridSearchCV, KFold
```
`GridSearchCV` is a module that takes in specified lists of parameters and runs through them to try to determine the best combination of parameters for a model. `KFold` is used for cross-validation purposes.

```python3
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
ada = AdaBoostClassifier()
search_grid = {'n_estimators':[100,250,500,750,1000,1500,2000],'learning_rate':[.001,0.01,.1]}
search = GridSearchCV(estimator=ada, param_grid=search_grid, scoring='accuracy', 
                      n_jobs=1, cv=crossvalidation)
```
This block of code creates a cross-validation variable, assigns our AdaBoost classifier, creates sets of parameters, and then uses `GridSearchCV` to find the combination to best tune our model. We will fit the `search` output variable to our test sets and see the results.

```python3
search.fit(x_test, y_test)
search.best_params_
```
```
{'learning_rate': 0.1, 'n_estimators': 500}
```
We can see that the `GridSearchCV` found that a learning rate of 0.1 with 500 estimators is our best combination. Let's try this out and see if our accuracy improves.

```python3
ada2 = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
ada2.fit(x_train, y_train)
ada2_pred = ada2.predict(x_test)
score = accuracy_score(y_test, ada2_pred) * 100
print("Accuracy using AdaBoost: ", round(score, 1), "%")
```
```
Accuracy using ada:  64.1 %
```
Our accuracy went up 3.6%, which is a good sign. For our final evaluation on the separate test playlist that was created, we will be looking for an accuracy of 75% or higher.

## Model Re-Testing

#### Import Test Data

In order to re-test our model, we will need to bring in the separate test playlist and create a dataframe in the same manner we did with the original 'Like' and 'Dislike' playlists.

```python3
# Bring in test playlist
test_ids = []
pl_id = 'spotify:playlist:3NpYLX125c2wLIvTwtfHZm'
offset = 0

while True:
    response = sp.playlist_tracks(pl_id,
                                  offset=offset,
                                  fields='items.track.id,total')
    test_ids.append(response['items'])
    offset = offset + len(response['items'])

    if len(response['items']) == 0:
        break

# Flatten list of lists of JSON
test_flatten = []
for sublist in test_ids:
    for item in sublist:
        test_flatten.append(item)
```
```python3
# Compile list of test track IDs
test_id_list = []
for i in range(0, len(test_flatten)):
    test_id_list.append(test_flatten[i]['track']['id'])
test_id_list = [x for x in test_id_list if x]

# Retrieve track characteristics
test_features = []
for i in range(0, len(test_id_list)):
    if not test_id_list[i]:
        continue
    else:
        test_features.append(sp.audio_features(test_id_list[i]))

# Flatten JSON list
test_features_flat = []
for sublist in test_features:
    for item in sublist:
        test_features_flat.append(item)
 ```
 ```python3
 # Create test dataframe
test_df = pd.DataFrame.from_records(test_features_flat)

# Retrieve song and artist names to add to dataframe
test_song_names = []
test_artists = []
for index, row in test_df.iterrows():
    try:
        response = sp.track(str(row['uri']))
        test_song_names.append(response['name'])
        test_artists.append(response['artists'][0]['name'])
    except SpotifyException as e:
        test_song_names.append('Unknown')
        test_artists.append('Unknown')

# Create 'song_name' and 'artist' columns
test_df['song_name'] = test_song_names
test_df['artist'] = test_artists
```
```python3
# Assign boolean like/dislike values
t = ([1] * 25) + ([0] * 25)

test_df['target'] = t
```

#### Testing/Evaluation

Now we can perform our re-testing and re-evaluation with our improved AdaBoost model.

```python3
from sklearn.metrics import confusion_matrix, classification_report

adab = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
adab.fit(x_train, y_train)
adab_pred = adab.predict(test_df[features])
score = accuracy_score(test_df['target'], adab_pred) * 100
print("Accuracy using AdaBoost: ", round(score, 1), "%")

# Add predictions to dataframe
test_df['pred'] = adab_pred

# Generate evaluation report
print(classification_report(test_df['target'], test_df['pred'], 
                            target_names=['Actual', 'Predicted']))

# Generate confusion matrix
confusion_matrix = pd.crosstab(test_df['target'], test_df['pred'], 
                               rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()
```

Here's the output for our classification report:
```python3
Accuracy using AdaBoost:  78.0 %
              precision    recall  f1-score   support

      Actual       0.94      0.60      0.73        25
   Predicted       0.71      0.96      0.81        25

    accuracy                           0.78        50
   macro avg       0.82      0.78      0.77        50
weighted avg       0.82      0.78      0.77        50
```

And here is the confusion matrix heatmap:

![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/plots/test_confusion_matrix.png?raw=true)

Our model was able to obtain an accuracy of 78% for our test set, which is above our target 75%! Looking at the confusion matrix, our model was able to correctly predict 24 out of 25 songs I liked, but only managed to get 15 out of 25 for songs I disliked. 


## AdaBoost Playlist Generator

Now that our improved model has been tested, we can use it to help create a playlist generator that will take in the top 20 tracks from a profile for the week, filter it through the AdaBoost model, and insert into the placeholder Spotify playlist.

#### Spotipy Instantiation

Because we are performing a different functionality to the previous scope, we need to establish a new Spotipy instance for the `user-top-read` scope, which will allow us to retrieve the top 20 tracks.

```python3
scope = "user-top-read"
red_uri = 'http://localhost:8080/callback'

# Connect and create Spotify instance
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp2 = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid, client_secret=secret,
                                                scope=scope, redirect_uri=red_uri,
                                                username='francescab13'))
```

#### Get Top 20 Tracks from Profile

Next, we'll extract a list of URI's for the top 20 tracks

```python3
# Create list of liked track IDs
liked_tracks = list(full_df['id'][full_df['target'] == 1])
top_tracks = sp2.current_user_top_tracks(limit=20, offset=0, time_range='medium_term')
top_20 = top_tracks['items']

# Get nested JSON
t20 = []
for i in top_20:
    t20.append(i['uri'])
```

#### Generate List of Recommended Songs

With our URI's retrieved, we can run this list through Spotipy's `recommendations` function, which can take in a list of seed artists, tracks, or genres and generate recommended songs. Only five tracks/artists can be supplied at a time, so this will need to be done in four groups. For each track passed through, two songs will be output, giving us a total of 40 recommended songs.

```python3
# Create list of recommended songs
# Note: Spotipy's 'recommendations' function can only take 5 track IDs at a time
# Five songs will be added to the playlist for every five track IDs analyzed
recs = []
i = 0
for i in range(0, len(t20)):
    if i == 0:
        recs.append(sp2.recommendations(seed_tracks=t20[0:5], limit=2))
        i += 5
    else:
        recs.append(sp2.recommendations(seed_tracks=t20[i:i+5], limit=2))
        i += 5
 ```
 
 We'll flatten this list and create a dataframe from it.
 
 ```python3
 rec_tracks = []
for i in recs:
    rec_tracks.append(i['tracks'])

# Flatten list of lists of JSON
rec_flatten = []
for sublist in rec_tracks:
    for item in sublist:
        rec_flatten.append(item)
rec_flatten[0:3]

# Create dataframe
rec_df = pd.DataFrame.from_records(rec_flatten)
```

#### Get Features for Recommended Songs

Before we can run these 40 songs through the AdaBoost model, we need to retrieve the audio features like we did for the like/dislike/test playlists. First, we'll obtain the audio features and flatten the JSON list.

```python3
# Compile list of test track IDs
rec_id_list = list(rec_df['uri'])

# Retrieve track characteristics
rec_features = []
for i in range(0, len(rec_id_list)):
    if not rec_id_list[i]:
        continue
    else:
        rec_features.append(sp.audio_features(rec_id_list[i]))

# Flatten JSON list
rec_features_flat = []
for sublist in rec_features:
    for item in sublist:
        rec_features_flat.append(item)
```

Then, we'll create a separate `feature_df` dataframe to hold the expanded `features` column from the `rec_df` dataframe. From there, we'll created a merged dataframe with all the necessary information to pass into the AdaBoost model.

```python3
# Add features column to dataframe
rec_df['features'] = rec_features_flat

# Separate out features into columns
feature_df = pd.json_normalize(rec_df['features'])
# Drop duplicate columns from feature_df
feature_df.drop(['type', 'id', 'uri', 'track_href'], axis=1, inplace=True)

# Concatenate feature dataframe with rec dataframe
merged_df = pd.concat([rec_df, feature_df], axis=1, sort=False)
```

#### Filter Recommended Songs through AdaBoost Model

Finally, we can take our merged dataframe and generate an output `final_df` dataframe which will only contain the recommended songs that the AdaBoost model predicted I would like.

```python3
adab = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
adab.fit(x_train, y_train)
adab_pred = adab.predict(merged_df[features])

# Add AdaBoost predictions to dataframe
merged_df['prediction'] = adab_pred

# Filter down dataframe to only positive predictions
final_df = merged_df[merged_df['prediction'] == 1]
```

### Add Filtered Songs to Playlist

The last thing to do is generate another new Spotipy instance to use the `playlist-modify-public` scope and insert the songs into the placeholder playlist.

```python3
scope = 'playlist-modify-public'
red_uri = 'http://localhost:8080/callback'

# Connect and create Spotify instance
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp3 = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid, client_secret=secret,
                                                scope=scope, redirect_uri=red_uri,
                                                username='francescab13'))

# Create Spotify playlist
sp3.user_playlist_add_tracks(user='francescab13',
                            playlist_id='3ltrx1uwJp4VciDPYBTG4r',
                            tracks=list(final_df['uri']))
```
