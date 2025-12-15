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
    // Do not clear results
  };

  const removeImage = (name) => {
    setImages(prev => prev.filter(img => img.name !== name));
    setResults(prev => prev.filter(res => res.filename !== name));
  };

  const analyzeImages = async () => {
    // Filter for images that don't have results yet
    const unanalyzedImages = images.filter(img => !results.some(r => r.filename === img.name));

    if (unanalyzedImages.length === 0) return;

    setLoading(true);

    const formData = new FormData();
    unanalyzedImages.forEach(img => {
      formData.append('images', img.file);
    });

    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5001';
      const response = await axios.post(`${apiUrl}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-type' }
      });

      // Merge new results with existing results
      setResults(prev => [...prev, ...response.data]);
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

  // Merge images with their results for display
  const activeData = images.map(img => {
    const result = results.find(r => r.filename === img.name);
    return result ? { ...img, ...result } : img;
  });

  /* Drag & Drop Handlers */
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      const newImages = files.map(file => ({
        file,
        preview: URL.createObjectURL(file),
        name: file.name
      }));
      setImages(prev => [...prev, ...newImages]);
      // Do not clear results
    }
  };

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
            <h1>Instagrammable</h1>
          </motion.div>
          <motion.p
            className="hero-subtitle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            Use our ensemble learning models to predict how viral and aesthetic your photo is.
          </motion.p>
        </motion.header>

        {/* UPLOAD SECTION */}
        <motion.div
          className="upload-container"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          <label
            className={`drop-zone ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="drop-zone-content">
              <div className="upload-icon-wrapper">
                <Upload size={32} color={isDragging ? "#d946ef" : "#8b5cf6"} />
              </div>
              <span className="drop-text-main">
                {isDragging ? "Drop to Upload" : "Upload Photos"}
              </span>
              <span className="drop-text-sub">
                {isDragging ? "Release your files here" : "Drag & drop or click to browse"}
              </span>
            </div>
            <input type="file" multiple accept="image/*" onChange={handleFileChange} hidden />

            {/* Ambient background glow for dropzone */}
            <div style={{
              position: 'absolute',
              inset: 0,
              background: isDragging
                ? 'radial-gradient(circle at center, rgba(217, 70, 239, 0.15), transparent 70%)'
                : 'radial-gradient(circle at center, rgba(139, 92, 246, 0.05), transparent 70%)',
              zIndex: 1,
              transition: 'background 0.3s ease'
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
                    onClick={(e) => { e.stopPropagation(); removeImage(item.name); }}
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

                  {/* Loading Scan Effect - Only for items being analyzed (no score yet) */}
                  {loading && item.average_score === undefined && (
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

                    {/* Detected Concepts */}
                    {item.concepts && item.concepts.length > 0 && (
                      <div className="concepts-section">
                        <div className="breakdown-title" style={{ marginTop: '16px' }}>
                          <Sparkles size={14} /> Aesthetic Vibe
                        </div>
                        <div className="concepts-tags">
                          {item.concepts.map((concept, i) => (
                            <span key={i} className="concept-tag">
                              {concept}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* AI Explanation */}
                    {item.explanation && (
                      <div className="explanation-section">
                        <div className="breakdown-title" style={{ marginTop: '16px' }}>
                          <Zap size={14} /> AI Verdict
                        </div>
                        <p className="explanation-text">
                          {item.explanation}
                        </p>
                      </div>
                    )}
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
