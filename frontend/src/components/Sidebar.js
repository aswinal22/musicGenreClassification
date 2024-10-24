import React from 'react';
import { Link } from 'react-router-dom'; // Import Link

const Sidebar = ({ genres, onSelectGenre }) => {
  return (
    <div className="sidebar">
      <h2>Genres</h2>
      <div className="sidebar-item">
        <Link to="/" onClick={() => onSelectGenre(null)}>All</Link>
      </div>
      {genres.map((genre) => (
        <div key={genre} className="sidebar-item">
          <Link to="/" onClick={() => onSelectGenre(genre)}>{genre}</Link>
        </div>
      ))}
      <div className="sidebar-item">
        <Link to="/favorites">Favorites</Link> {/* Link to Favorites */}
      </div>
      <div className="sidebar-item">
        <Link to="/recycle-bin">Recycle Bin</Link> {/* Link to Recycle Bin */}
      </div>
    </div>
  );
};

export default Sidebar;
