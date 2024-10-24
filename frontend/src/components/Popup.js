import React, { useEffect } from 'react';
import './Popup.css';

const Popup = ({ message, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000); // Close after 3 seconds
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className="popup">
      <div className="popup-content">
        {message}
      </div>
    </div>
  );
};

export default Popup;