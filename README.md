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

```python3
fig, ax = plt.subplots(len(trait_cols), figsize=(16,12))

for i, col_val in enumerate(trait_cols):

    sns.distplot(like_df[col_val], hist=True, ax=ax[i])
    ax[i].set_title('Freq dist '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    ax[i].set_ylabel('Count', fontsize=8)
```
![alt text](https://github.com/francescabeller/MSDS-696-Practicum/blob/master/like_dist_plots.png?raw=true)

