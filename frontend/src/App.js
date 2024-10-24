import React, { useState, useEffect } from 'react';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import MainView from './components/MainView';
import Player from './components/Player';
import Favorites from './components/Favorites';
import RecycleBin from './components/RecycleBin';
import Popup from './components/Popup'; // Import Popup
import axios from 'axios';

// Sample song data
const sampleSongs = [
  { id: 1, title: 'Song A', artist: 'Artist 1', genre: 'Pop', albumArt: 'https://via.placeholder.com/150' },
  { id: 2, title: 'Song B', artist: 'Artist 2', genre: 'Rock', albumArt: 'https://via.placeholder.com/150' },
  { id: 3, title: 'Song C', artist: 'Artist 3', genre: 'Jazz', albumArt: 'https://via.placeholder.com/150' },
  { id: 4, title: 'Song D', artist: 'Artist 4', genre: 'Hip Hop', albumArt: 'https://via.placeholder.com/150' },
  { id: 5, title: 'Song E', artist: 'Artist 5', genre: 'Classical', albumArt: 'https://via.placeholder.com/150' },
  { id: 6, title: 'Song F', artist: 'Artist 6', genre: 'Electronic', albumArt: 'https://via.placeholder.com/150' },
];

// Predefined genres
const predefinedGenres = [
  'blues',
  'classical',
  'country',
  'disco',
  'hiphop',
  'jazz',
  'metal',
  'pop',
  'reggae',
  'rock',
];

function App() {
  const [songs, setSongs] = useState(sampleSongs); // Use sampleSongs as initial state
  const [currentGenre, setCurrentGenre] = useState(null);
  const [currentSong, setCurrentSong] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [favorites, setFavorites] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [recycleBinSongs, setRecycleBinSongs] = useState([]);
  const [popupMessage, setPopupMessage] = useState(''); // State for popup message

  useEffect(() => {
    const fetchSongs = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/songs');
        console.log('Fetched songs:', response.data); // Add this line to debug
        setSongs(response.data);
      } catch (error) {
        console.error('Error fetching songs:', error);
      }
    };

    fetchSongs();
    fetchRecycleBinSongs(); // Fetch recycle bin songs on initial load
  }, []);

  const fetchRecycleBinSongs = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/recycle-bin');
      setRecycleBinSongs(response.data);
    } catch (error) {
      console.error('Error fetching recycle bin songs:', error);
    }
  };

  const handleRestoreSong = async (id) => {
    try {
      await axios.post(`http://localhost:5000/api/recycle-bin/${id}/restore`);
      fetchRecycleBinSongs();
      const response = await axios.get('http://localhost:5000/api/songs');
      setSongs(response.data);
    } catch (error) {
      console.error('Error restoring song:', error);
    }
  };

  const handleDeletePermanently = async (id) => {
    console.log(`Attempting to permanently delete song with id: ${id}`); // Add this line to debug
    try {
      await axios.delete(`http://localhost:5000/api/recycle-bin/${id}`);
      fetchRecycleBinSongs(); // Fetch recycle bin songs after deleting a song permanently
      setPopupMessage('Song permanently deleted'); // Set popup message
    } catch (error) {
      console.error('Error deleting song permanently:', error);
    }
  };

  const handleSelectGenre = (genre) => {
    setCurrentGenre(genre);
  };

  const handleSelectSong = (song) => {
    setCurrentSong(song);
    setIsPlaying(true);
  };

  const handlePlayPause = () => {
    setIsPlaying(prev => !prev);
  };

  const handleToggleFavorite = (id) => {
    setFavorites(prevFavorites =>
      prevFavorites.includes(id) ? prevFavorites.filter(favId => favId !== id) : [...prevFavorites, id]
    );
  };

  const handleDeleteSong = async (id) => {
    try {
      await axios.delete(`http://localhost:5000/api/songs/${id}`);
      setSongs(prevSongs => prevSongs.filter(song => song._id !== id));
      fetchRecycleBinSongs(); // Fetch recycle bin songs after deleting a song
      setPopupMessage('Song moved to recycle bin'); // Set popup message
    } catch (error) {
      console.error('Error deleting song:', error);
    }
  };

  const handleUpload = async (songData) => {
    const formData = new FormData();
    formData.append('title', songData.title);
    formData.append('artist', songData.artist);
    formData.append('file', songData.file);

    try {
      const response = await axios.post('http://localhost:5000/api/songs', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setSongs(prevSongs => [...prevSongs, response.data]);
    } catch (error) {
      console.error('Error uploading song:', error);
    }
  };

  const toggleDarkMode = () => {
    setIsDarkMode(prev => !prev);
  };

  const handlePreviousSong = () => {
    if (!currentSong) return;
    const currentIndex = songs.findIndex(song => song._id === currentSong._id);
    if (currentIndex > 0) {
      setCurrentSong(songs[currentIndex - 1]);
      setIsPlaying(true);
    }
  };

  const handleNextSong = () => {
    if (!currentSong) return;
    const currentGenreSongs = songs.filter(song => song.genre === currentSong.genre);
    const currentIndex = currentGenreSongs.findIndex(song => song._id === currentSong._id);
    if (currentIndex < currentGenreSongs.length - 1) {
      setCurrentSong(currentGenreSongs[currentIndex + 1]);
      setIsPlaying(true);
    }
  };

  return (
    <Router>
      <div className={`App ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
        <div className="spotify-body">
          <Sidebar 
            genres={predefinedGenres} 
            onSelectGenre={handleSelectGenre} 
          />
          <Routes>
            <Route path="/" element={
              <MainView 
                songs={songs.filter(song => !currentGenre || song.genre === currentGenre)} 
                onUpload={handleUpload} 
                onSelectSong={handleSelectSong} 
                onToggleFavorite={handleToggleFavorite}
                onDeleteSong={handleDeleteSong}
                favorites={favorites}
              />
            } />
            <Route path="/favorites" element={
              <Favorites songs={songs} favorites={favorites} onSelectSong={handleSelectSong} />
            } />
            <Route path="/recycle-bin" element={
              <RecycleBin songs={recycleBinSongs} onRestoreSong={handleRestoreSong} onDeletePermanently={handleDeletePermanently} />
            } />
          </Routes>
        </div>
        <Player 
          currentSong={currentSong} 
          onPlayPause={handlePlayPause} 
          isPlaying={isPlaying} 
          onPrevious={handlePreviousSong} 
          onNext={handleNextSong} 
          onToggleFavorite={handleToggleFavorite}
        />
        <button onClick={toggleDarkMode} className="toggle-button">
          {isDarkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
        {popupMessage && <Popup message={popupMessage} onClose={() => setPopupMessage('')} />} {/* Show popup */}
      </div>
    </Router>
  );
}

export default App;
