import { useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { ErrorBox } from '../components/ui'

export default function Login() {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('demo@example.com')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to="/" replace />

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await login(email, password)
      navigate(location.state?.from?.pathname || '/', { replace: true })
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-panel">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">◆</span>
          <div>
            <div className="brand-name">RecLab</div>
            <div className="brand-sub">Multimodal recommender</div>
          </div>
        </div>
        <h1>Sign in</h1>
        <p className="muted">
          Two-tower retrieval over CLIP image + text features, FAISS candidate search and a LightGBM
          LambdaRank reranker, trained on Amazon Reviews 2023.
        </p>
        <form onSubmit={submit} className="form">
          <label>
            <span>Email</span>
            <input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label>
            <span>Password</span>
            <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required autoFocus />
          </label>
          <ErrorBox error={error} />
          <button className="button primary block" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
        </form>
      </div>
    </div>
  )
}
