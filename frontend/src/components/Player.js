import React, { useRef, useEffect } from 'react';

const Player = ({ currentSong, onPlayPause, isPlaying, onPrevious, onNext, onToggleFavorite }) => {
  const audioRef = useRef(null);

  useEffect(() => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.play();
      } else {
        audioRef.current.pause();
      }
    }
  }, [isPlaying, currentSong]);

  return (
    <div className="player">
      {currentSong ? (
        <div className="player-content">
          <div className="album-art-container">
            <img
              src={currentSong.albumArt}
              alt={`${currentSong.title} Album Art`}
              className={`album-art ${isPlaying ? 'flip' : ''}`}
            />
          </div>
          <h3>Now Playing: {currentSong.title} by {currentSong.artist}</h3>
          <audio ref={audioRef} src={`http://localhost:5000/${currentSong.filePath}`} />
          <div className="player-controls">
            <button className="control-button" onClick={onPrevious}>⏮️</button>
            <button className="play-pause-button" onClick={onPlayPause}>
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button className="control-button" onClick={onNext}>⏭️</button>
            <button className="favorite-button" onClick={() => onToggleFavorite(currentSong._id)}>❤️</button>
          </div>
        </div>
      ) : (
        <div>No song selected</div>
      )}
    </div>
  );
};

export default Player;
