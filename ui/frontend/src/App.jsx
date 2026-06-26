import { Routes, Route, NavLink } from 'react-router-dom'
import NlLatentsDetailPage from './pages/NlLatentsDetailPage.jsx'
import NlLatentsPage from './pages/NlLatentsPage.jsx'
import ProvidersPage from './pages/ProvidersPage.jsx'
import './App.css'

function App() {
  return (
    <div className="app">
      <nav className="sidebar">
        <div className="sidebar-header">
          <h1 className="logo">dr-llm</h1>
          <span className="logo-sub">ui tools</span>
        </div>
        <div className="nav-section">
          <span className="nav-section-label">Explore</span>
          <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <svg aria-hidden="true" role="presentation" focusable="false" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="2" width="5" height="5" rx="1" />
              <rect x="9" y="2" width="5" height="5" rx="1" />
              <rect x="2" y="9" width="5" height="5" rx="1" />
              <rect x="9" y="9" width="5" height="5" rx="1" />
            </svg>
            Providers & Models
          </NavLink>
          <NavLink to="/nl-latents" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <svg aria-hidden="true" role="presentation" focusable="false" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12.5V3.5" />
              <path d="M8 12.5V3.5" />
              <path d="M13 12.5V3.5" />
              <path d="M2 5.5H14" />
              <path d="M2 10.5H14" />
            </svg>
            nl-latents
          </NavLink>
        </div>
      </nav>
      <main className="content">
        <Routes>
          <Route path="/" element={<ProvidersPage />} />
          <Route path="/nl-latents" element={<NlLatentsPage />} />
          <Route path="/nl-latents/samples/:sampleId" element={<NlLatentsDetailPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
