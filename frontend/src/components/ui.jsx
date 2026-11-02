import { useState } from 'react'
import { imageUrl } from '../api'
import { compact } from '../format'

export function Stat({ label, value, hint, tone }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone ? `tone-${tone}` : ''}`}>{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  )
}

export function Empty({ title, children }) {
  return (
    <div className="card empty">
      {title && <div className="empty-title">{title}</div>}
      {children && <div className="muted">{children}</div>}
    </div>
  )
}

export function ErrorBox({ error }) {
  if (!error) return null
  return <div className="error-box" role="alert">{error.message || String(error)}</div>
}

export function Spinner({ label = 'Loading…' }) {
  return <div className="spinner-row"><span className="spinner" aria-hidden="true" />{label}</div>
}

export function Segmented({ value, onChange, options, label }) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={value === o.value ? 'active' : ''}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function ProductImage({ product, className = '' }) {
  const [src, setSrc] = useState(() => imageUrl(product))
  const [failed, setFailed] = useState(false)
  if (failed) return <div className={`product-image placeholder ${className}`}>{product.category}</div>
  return (
    <img
      className={`product-image ${className}`}
      src={src}
      alt=""
      loading="lazy"
      onError={() => {
        if (product.image_url && src !== product.image_url) setSrc(product.image_url)
        else setFailed(true)
      }}
    />
  )
}

export function Rating({ value, count }) {
  if (value == null) return <span className="muted">No ratings</span>
  return (
    <span className="rating" title={`${value} out of 5`}>
      <span aria-hidden="true">★</span> {Number(value).toFixed(1)}
      {count != null && <span className="muted"> ({compact(count)})</span>}
    </span>
  )
}
