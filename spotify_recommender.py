###############################################
########### Spotify Authentication ############
###############################################

# Import libraries
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import pandas as pd

# Set account info
cid = '81fee852cceb4259910e7d2ff78493c3'
secret = 'ad4360215d7641ee809275cc5cdd4a6c'
username = 'francescab13'

# Connect and create Spotify instance
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

###############################################
############### Data Gathering ################
###############################################

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


# Get audio features for liked songs
good_id_list = []
for i in range(0, len(good_flatten)):
    good_id_list.append(good_flatten[i]['track']['id'])
good_id_list = [x for x in good_id_list if x]

good_features = []
for i in range(0, len(good_id_list)):
    if not good_id_list[i]:
        continue
    else:
        good_features.append(sp.audio_features(good_id_list[i]))
good_features_flat = []
for sublist in good_features:
    for item in sublist:
        good_features_flat.append(item)


# Get audio features for disliked songs
bad_id_list = []
for i in range(0, len(bad_flatten)):
    bad_id_list.append(bad_flatten[i]['track']['id'])
bad_id_list = [x for x in bad_id_list if x]

bad_features = []
for i in range(0, len(bad_id_list)):
    if not bad_id_list[i]:
        continue
    else:
        bad_features.append(sp.audio_features(bad_id_list[i]))
bad_features_flat = []
for sublist in bad_features:
    for item in sublist:
        bad_features_flat.append(item)

# Create 'Like' and 'Dislike' dataframes
like_df = pd.DataFrame.from_records(good_features_flat)

good_uris = list(like_df['uri'])
good_song_names = []
for i in range(0, len(good_id_list)):
    good_song_names.append(sp.track(good_uris[i]))

good_song_names = []
good_artists = []
for i in range(0, 5):
    response = sp.track(good_uris[i])
    good_song_names.append(response['name'])
    good_artists.append(response['artists'][0]['name'])

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

like_df['song_name'] = good_song_names
like_df['artist'] = good_artists

len(like_df[like_df['song_name'] == 'Unknown'])

# Get list of songs that returned error and re-search
missed_songs = list(like_df['uri'][like_df['song_name'] == 'Unknown'])

missed_names = []
for i in range(0, len(missed_songs)):
    try:
        response = sp.track(missed_songs[i])
        missed_names.append(response['name'])
    except SpotifyException as e:
        missed_names.append('UNKNOWN')

# Create dataframe for missing songs
missed_df = pd.DataFrame({'uri': missed_songs, 'song_name': missed_names})

# Replace final missed track
missing = missed_df['uri'][missed_df['song_name'] == 'UNKNOWN']
sp.track(missing)
missed_df['song_name'].replace('UNKNOWN', 'Polarize', inplace=True)

# Get list of artists that returned error and re-search
missed_artists = list(like_df['uri'][like_df['artist'] == 'Unknown'])

missed_art = []
for i in range(0, len(missed_artists)):
    try:
        response = sp.track(missed_artists[i])
        missed_art.append(response['artists'][0]['name'])
    except SpotifyException as e:
        missed_names.append('UNKNOWN')

# Add missing artist list to 'missing' dataframe
missed_df['artist'] = missed_art

# Merge missing songs/artists with original 'like' dataframe
s = missed_df.set_index('uri')['song_name'].to_dict()
a = missed_df.set_index('uri')['artist'].to_dict()
v = like_df.filter(items='song_name')
like_df[v.columns] = v.replace(s)
v2 = like_df.filter(items='artist')
like_df[v2.columns] = v2.replace(a)
