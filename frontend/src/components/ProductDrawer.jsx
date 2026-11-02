import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { useShopper } from '../shopper'
import { CartIcon, CloseIcon } from './icons'
import { price } from '../format'
import { useAsync } from '../useAsync'
import { ErrorBox, ProductImage, Rating, Spinner } from './ui'

// Product detail with "more like this" from the FAISS item index. Opening a
// product records a click for the active shopper; the buttons record
// stronger-intent events. Each event immediately changes recommendations.
export default function ProductDrawer({ product, requestId, onClose, onSelect }) {
  const { shopper, track } = useShopper()
  // Parents render this with key={product.id}, so local state starts fresh per product.
  const { data: detail, error: loadError } = useAsync(() => api.product(product.id), [product.id])
  const [actionError, setActionError] = useState(null)
  const [done, setDone] = useState({})
  const error = loadError || actionError

  // Record the click once per opened product (StrictMode re-runs effects in dev).
  const tracked = useRef(false)
  useEffect(() => {
    if (tracked.current || !shopper) return
    tracked.current = true
    track(product.id, 'click', requestId).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [product.id])

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const act = async (type) => {
    try {
      await track(product.id, type, requestId)
      setDone((d) => ({ ...d, [type]: true }))
    } catch (e) {
      setActionError(e)
    }
  }

  const p = detail?.product || product
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={p.title} onClick={(e) => e.stopPropagation()}>
        <button type="button" className="icon-button drawer-close" onClick={onClose} aria-label="Close"><CloseIcon /></button>
        <div className="drawer-hero">
          <ProductImage product={p} className="large" />
        </div>
        <div className="drawer-body">
          <div className="product-meta">
            <span className="badge">{p.category}</span>
            {p.store && <span className="muted small">by {p.store}</span>}
          </div>
          <h2>{p.title}</h2>
          <div className="row gap">
            <Rating value={p.average_rating} count={p.rating_number} />
            {price(p.price) && <strong>{price(p.price)}</strong>}
          </div>
          {shopper ? (
            <div className="row gap actions">
              <button type="button" className="button" disabled={done.add_to_cart} onClick={() => act('add_to_cart')}>
                <CartIcon /> {done.add_to_cart ? 'Added' : 'Add to cart'}
              </button>
              <button type="button" className="button primary" disabled={done.purchase} onClick={() => act('purchase')}>
                {done.purchase ? 'Purchased' : 'Buy now'}
              </button>
            </div>
          ) : (
            <p className="muted small">Choose a shopper to record interactions.</p>
          )}
          <ErrorBox error={error} />
          {detail?.product.features && <p className="description">{detail.product.features.slice(0, 600)}</p>}

          <h3 className="section-title">More like this</h3>
          <p className="muted small">Nearest neighbours in the learned item-embedding space (FAISS HNSW).</p>
          {!detail ? <Spinner /> : detail.similar.length === 0 ? (
            <p className="muted">No similar items available.</p>
          ) : (
            <div className="mini-grid">
              {detail.similar.map(({ product: s, score }) => (
                <button type="button" key={s.id} className="mini-card" onClick={() => onSelect(s)} title={s.title}>
                  <ProductImage product={s} />
                  <span className="mini-title">{s.title}</span>
                  <span className="muted small">sim {score.toFixed(2)}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}
