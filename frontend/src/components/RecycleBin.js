import React from 'react';

const RecycleBin = ({ songs, onRestoreSong, onDeletePermanently }) => (
  <div className="main-view">
    <h2>Recycle Bin</h2>
    {songs.length === 0 ? (
      <p>No songs in the recycle bin.</p>
    ) : (
      songs.map((song) => (
        <div key={song._id} className="song-item">
          {song.title} by {song.artist}
          <button
            onClick={() => {
              console.log(`Restoring song with id: ${song._id}`); // Add this line to debug
              onRestoreSong(song._id);
            }}
            className="restore-button"
          >
            ♻️
          </button>
          <button
            onClick={() => {
              console.log(`Deleting song permanently with id: ${song._id}`); // Add this line to debug
              onDeletePermanently(song._id);
            }}
            className="delete-button"
          >
            🗑️
          </button>
        </div>
      ))
    )}
  </div>
);

export default RecycleBin;