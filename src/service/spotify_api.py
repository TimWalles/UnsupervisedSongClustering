import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth


class SpotifyApi:
    def __token(self):
        return SpotifyOAuth(
            client_id=os.getenv('CLIENT_ID'),
            client_secret=os.getenv('CLIENT_SECRET'),
            redirect_uri=os.getenv('REDIRECTION_URL'),
            scope='playlist-modify-public',
            username=os.getenv('USER_ID'),
        )

    def __get_handler(self):
        return spotipy.Spotify(auth_manager=self.__token())

    def upload_new_playlist(
        self,
        handler: spotipy.Spotify,
        playlist_name: str,
        uris: list[str],
    ) -> str:
        playlist_id = handler.user_playlist_create(
            user=os.getenv('USER_ID'),
            name=playlist_name,
        )['id']
        handler.user_playlist_add_tracks(
            user=os.getenv('USER_ID'),
            playlist_id=playlist_id,
            tracks=uris,
        )
        return playlist_id

    def update_playlist(
        self,
        handler: spotipy.Spotify,
        playlist_id: str,
        uris: list[str],
    ) -> str:
        handler.user_playlist_replace_tracks(
            user=os.getenv('USER_ID'),
            playlist_id=playlist_id,
            tracks=uris,
        )
        return playlist_id

    def upsert_spotify_playlist(
        self,
        song_ids: list,
        playlist_id: str | None = None,
        playlist_name: str | None = None,
    ):
        handler = self.__get_handler()
        uris = [f'spotify:track:{str(song_id).strip()}' for song_id in song_ids]
        if playlist_id:
            return self.update_playlist(handler, playlist_id, uris)
        elif playlist_name:
            return self.upload_new_playlist(handler, playlist_name, uris)
