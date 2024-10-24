import React, { useState } from 'react';

const MainView = ({ songs, onUpload, onSelectSong, onToggleFavorite, onDeleteSong, favorites }) => {
  const [title, setTitle] = useState('');
  const [artist, setArtist] = useState('');
  const [file, setFile] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file) {
      onUpload({ title, artist, file });
      setTitle('');
      setArtist('');
      setFile(null);
    }
    window.location.reload();
  };

  return (
    <div className="main-view">
      <h2>All Songs</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="input-field title-input"
          required
        />
        <input
          type="text"
          placeholder="Artist"
          value={artist}
          onChange={(e) => setArtist(e.target.value)}
          className="input-field artist-input"
          required
        />
        <input
          type="file"
          accept="audio/mp3"
          onChange={(e) => setFile(e.target.files[0])}
          className="file-input"
          required
        />
        <button className='upload-button' type="submit">Upload</button>
      </form>
      <div className="song-list">
        {songs.map((song) => (
          <div key={song._id} onClick={() => onSelectSong(song)} className="song-item">
            {song.title} by {song.artist}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleFavorite(song._id);
              }}
              className="favorite-button"
            >
              {favorites.includes(song._id) ? '★' : '☆'}
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSong(song._id);
              }}
              className="delete-button"
            >
              🗑️
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MainView;
