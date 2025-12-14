import React, { useState } from 'react';
import axios from 'axios';
import { Upload, X, Loader2, Sparkles, Zap, ScanLine } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './App.css';

function App() {
  const [images, setImages] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map(file => ({
      file,
      preview: URL.createObjectURL(file),
      name: file.name
    }));
    setImages(prev => [...prev, ...newImages]);
    setResults([]);
  };

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
    if (results.length > 0) {
        setResults(prev => prev.filter((_, i) => i !== index));
    }
  };

  const analyzeImages = async () => {
    if (images.length === 0) return;
    setLoading(true);
    setResults([]);

    const formData = new FormData();
    images.forEach(img => {
      formData.append('images', img.file);
    });

    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5002';
      const response = await axios.post(`${apiUrl}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-type' }
      });
      
      const mergedResults = images.map(img => {
        const data = response.data.find(r => r.filename === img.name);
        return { ...img, ...data };
      });
      
      setResults(mergedResults);
    } catch (error) {
      console.error("Error analyzing images:", error);
      alert("Failed to connect to the backend server.");
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 4.0) return '#4ade80'; 
    if (score >= 2.5) return '#facc15'; 
    return '#f87171'; 
  };

  const activeData = results.length > 0 ? results : images;

  return (
    <>
      <div className="app-bg-wrapper">
        <div className="grid-overlay" />
      </div>
      
      <div className="app-container">
        
        {/* HEADER */}
        <motion.header 
          className="hero"
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.8 }}
          >
            <h1>InstaRate AI</h1>
          </motion.div>
          <motion.p 
            className="hero-subtitle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            Deploy our ensemble neural networks to predict your social viral potential with precision.
          </motion.p>
        </motion.header>

        {/* UPLOAD SECTION */}
        <motion.div 
          className="upload-container"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          <label className="drop-zone">
            <div className="drop-zone-content">
              <div className="upload-icon-wrapper">
                <Upload size={32} color="#8b5cf6" />
              </div>
              <span className="drop-text-main">Upload Photos</span>
              <span className="drop-text-sub">Drag & drop or click to browse</span>
            </div>
            <input type="file" multiple accept="image/*" onChange={handleFileChange} hidden />
            
            {/* Ambient background glow for dropzone */}
            <div style={{
                position: 'absolute',
                inset: 0,
                background: 'radial-gradient(circle at center, rgba(139, 92, 246, 0.05), transparent 70%)',
                zIndex: 1
            }} />
          </label>
            
          <AnimatePresence mode="wait">
            {images.length > 0 && (
              <motion.div 
                className="analyze-btn-wrapper"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <button 
                    className="primary-btn" 
                    onClick={analyzeImages} 
                    disabled={loading}
                >
                  <div className="btn-glow" />
                  {loading ? (
                    <>
                        <Loader2 className="spin" size={20} />
                        Processing...
                    </>
                  ) : (
                    <>
                        <Sparkles size={20} />
                        Analyze Virality
                    </>
                  )}
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* RESULTS GRID */}
        <motion.div 
          className="masonry-grid"
          layout
        >
          <AnimatePresence>
            {activeData.map((item, idx) => (
              <motion.div 
                key={item.preview}
                className="glass-card"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
                transition={{ delay: idx * 0.05, duration: 0.5 }}
                layout
              >
                <div className="card-img-container">
                  <img src={item.preview} alt="preview" />
                  
                  {/* Remove Button */}
                  <motion.button 
                    className="remove-action"
                    onClick={(e) => { e.stopPropagation(); removeImage(idx); }}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <X size={18} />
                  </motion.button>

                  {/* Main Score Badge Overlay */}
                  {item.average_score !== undefined && (
                    <motion.div 
                        className="card-score-badge"
                        initial={{ x: 20, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        transition={{ delay: 0.2 }}
                    >
                        <Zap size={16} color={getScoreColor(item.average_score)} fill={getScoreColor(item.average_score)} />
                        <span className="score-value">{item.average_score.toFixed(1)}</span>
                        <span className="score-max">/ 5.0</span>
                    </motion.div>
                  )}

                  {/* Loading Scan Effect */}
                  {loading && (
                    <motion.div
                        initial={{ top: 0 }}
                        animate={{ top: "100%" }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                        style={{
                            position: 'absolute',
                            left: 0,
                            right: 0,
                            height: '2px',
                            background: '#8b5cf6',
                            boxShadow: '0 0 10px #8b5cf6',
                            zIndex: 10
                        }}
                    />
                  )}
                </div>

                {/* Analysis Details */}
                {item.breakdown && (
                  <motion.div 
                    className="card-content"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4 }}
                  >
                    <div className="breakdown-title">
                        <ScanLine size={14} /> Neural Analysis
                    </div>
                    
                    {Object.entries(item.breakdown).map(([model, score], i) => (
                      <div key={model}>
                        <div className="stat-row">
                          <span className="stat-label">{model}</span>
                          <span className="stat-value" style={{ color: getScoreColor(score) }}>{score}</span>
                        </div>
                        <div className="bar-bg">
                          <motion.div 
                            className="bar-fill"
                            initial={{ width: 0 }}
                            animate={{ width: `${(score / 5) * 100}%` }}
                            transition={{ duration: 1, delay: 0.5 + (i * 0.1), ease: "easeOut" }}
                            style={{ 
                              background: score > 3.5 
                                ? 'linear-gradient(90deg, #8b5cf6, #d946ef)' 
                                : score > 2.5 ? '#facc15' : '#94a3b8' 
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </motion.div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      </div>
    </>
  );
}

export default App;