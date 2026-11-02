import { useEffect, useState } from 'react'
import { api } from '../api'
import { useShopper } from '../shopper'
import { CloseIcon, PlusIcon } from './icons'
import { ErrorBox, Spinner } from './ui'

export default function ShopperPicker({ onClose }) {
  const { shopper, select, createNew } = useShopper()
  const [query, setQuery] = useState('')
  const [rows, setRows] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    const timer = setTimeout(() => {
      api.customers({ q: query || undefined, limit: 30 }).then(setRows).catch(setError)
    }, 200)
    return () => clearTimeout(timer)
  }, [query])

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const choose = async (id) => {
    try {
      await select(id)
      onClose()
    } catch (e) {
      setError(e)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="Choose a shopper" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <h2>Choose a shopper</h2>
            <p className="muted">Real Amazon customers from the dataset, or start a brand-new shopper with no history.</p>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close"><CloseIcon /></button>
        </div>
        <div className="modal-tools">
          <input
            autoFocus
            className="input"
            placeholder="Search by name or customer id"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="button" className="button primary" onClick={async () => { await createNew(); onClose() }}>
            <PlusIcon /> New shopper
          </button>
        </div>
        <ErrorBox error={error} />
        {!rows ? <Spinner /> : (
          <ul className="shopper-list">
            {rows.map((r) => (
              <li key={r.id}>
                <button type="button" className={`shopper-row ${shopper?.id === r.id ? 'active' : ''}`} onClick={() => choose(r.id)}>
                  <span className="avatar" aria-hidden="true">{r.display_name.split(' ').pop().slice(-2)}</span>
                  <span className="grow">
                    <span className="row-title">{r.display_name}</span>
                    <span className="muted small">{r.top_category || 'No history yet'}</span>
                  </span>
                  {r.source === 'live' && <span className="badge accent">live</span>}
                  <span className="muted small">{r.interaction_count} events</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
