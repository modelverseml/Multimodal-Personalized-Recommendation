import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth'
import { useShopper } from '../shopper'
import { ChartIcon, GridIcon, PulseIcon, SparkIcon, UserIcon } from './icons'
import ShopperPicker from './ShopperPicker'

const NAV = [
  { to: '/', label: 'For you', icon: SparkIcon, end: true },
  { to: '/shop', label: 'Catalog', icon: GridIcon },
  { to: '/evaluation', label: 'Evaluation', icon: ChartIcon },
  { to: '/monitoring', label: 'Monitoring', icon: PulseIcon },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const { shopper } = useShopper()
  const [picking, setPicking] = useState(false)

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">◆</span>
          <div>
            <div className="brand-name">RecLab</div>
            <div className="brand-sub">Multimodal recommender</div>
          </div>
        </div>
        <nav>
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Icon />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="muted small">{user?.email}</div>
          <button type="button" className="link-button" onClick={logout}>Sign out</button>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <button type="button" className="shopper-pill" onClick={() => setPicking(true)}>
            <UserIcon />
            {shopper ? (
              <span>
                Shopping as <strong>{shopper.display_name}</strong>
                <span className="muted"> · {shopper.interaction_count} interactions</span>
              </span>
            ) : (
              <span>Choose a shopper</span>
            )}
          </button>
        </header>
        <Outlet context={{ openPicker: () => setPicking(true) }} />
      </div>
      {picking && <ShopperPicker onClose={() => setPicking(false)} />}
    </div>
  )
}
