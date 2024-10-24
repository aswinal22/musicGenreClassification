import React from 'react';
import '../App.css';

const Favorites = ({ songs, favorites, onSelectSong }) => (
  <div className="main-view">
    <h2>Favorites</h2>
    {favorites.length === 0 ? (
      <p>No favorites added yet.</p> // Message when no favorites
    ) : (
      songs.filter(song => favorites.includes(song._id)).map((song) => (
        <div key={song._id} onClick={() => onSelectSong(song)} className="song-item">
          {song.title} by {song.artist} ({song.genre})
        </div>
      ))
    )}
  </div>
);

export default Favorites;
