import React from 'react';
import '../App.css';

const AllSongs = ({ songs, onSelectSong }) => (
  <div>
    <h2>All Songs</h2>
    {songs.map((song) => (
      <div key={song.id} onClick={() => onSelectSong(song)} className="song-item">
        {song.title} by {song.artist} ({song.genre})
      </div>
    ))}
  </div>
);

export default AllSongs;
