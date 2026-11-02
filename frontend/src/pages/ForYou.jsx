import { useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import { api } from '../api'
import { RefreshIcon } from '../components/icons'
import ProductCard from '../components/ProductCard'
import ProductDrawer from '../components/ProductDrawer'
import { Empty, ErrorBox, ProductImage, Segmented, Spinner } from '../components/ui'
import { ms } from '../format'
import { useShopper } from '../shopper'
import { useAsync } from '../useAsync'

const STRATEGY_LABEL = {
  'two_tower+ltr': 'Two-tower retrieval → LambdaRank',
  two_tower: 'Two-tower retrieval only',
  trending: 'Cold start: trending items',
}

export default function ForYou() {
  const { shopper, version, createNew } = useShopper()
  const { openPicker } = useOutletContext()
  const [mode, setMode] = useState('rerank')
  const [open, setOpen] = useState(null)
  // Refetch whenever the shopper changes, the mode flips, or a new event is recorded.
  const shopperId = shopper?.id
  const { data: recs, error, loading, reload: load } = useAsync(
    () => (shopperId ? api.recommendations(shopperId, { k: 12, rerank: mode === 'rerank' }) : Promise.resolve(null)),
    [shopperId, mode, version],
  )

  if (!shopper) {
    return (
      <div className="page">
        <div className="page-head"><h1>For you</h1></div>
        <Empty title="Pick a shopper to see personalised recommendations">
          <p>Browse as a real customer from the dataset, or start a new shopper and watch the
            recommendations adapt after every click; there is no retraining involved.</p>
          <div className="row gap center">
            <button type="button" className="button primary" onClick={openPicker}>Choose a shopper</button>
            <button type="button" className="button" onClick={createNew}>Start a new shopper</button>
          </div>
        </Empty>
      </div>
    )
  }

  const history = shopper.history.slice(0, 14)
  const t = recs?.timings || {}

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>For you</h1>
          <p className="muted">
            Recommendations for <strong>{shopper.display_name}</strong> from their {shopper.interaction_count} interactions.
            Open, add or buy an item and the list updates from the new history straight away.
          </p>
        </div>
        <div className="row gap">
          <Segmented
            label="Ranking stage"
            value={mode}
            onChange={setMode}
            options={[{ value: 'rerank', label: 'With reranker' }, { value: 'retrieval', label: 'Retrieval only' }]}
          />
          <button type="button" className="icon-button" onClick={load} aria-label="Refresh"><RefreshIcon /></button>
        </div>
      </div>

      <section className="card history-card">
        <div className="history-head">
          <h2 className="section-title">Recent history</h2>
          <div className="mix">
            {Object.entries(shopper.category_mix).slice(0, 4).map(([cat, share]) => (
              <span key={cat} className="mix-chip">{cat} <span className="muted">{Math.round(share * 100)}%</span></span>
            ))}
          </div>
        </div>
        {history.length === 0 ? (
          <p className="muted">No interactions yet, so this shopper gets trending items. Open anything in the catalog to personalise.</p>
        ) : (
          <div className="history-strip">
            {history.map((h, i) => (
              <button type="button" key={`${h.product.id}-${i}`} className="history-item" onClick={() => setOpen({ product: h.product })} title={h.product.title}>
                <ProductImage product={h.product} />
                <span className={`event-dot ${h.event_type}`} title={h.event_type} />
                <span className="history-label">{h.event_type.replace('_', ' ')}</span>
              </button>
            ))}
          </div>
        )}
      </section>

      {recs && (
        <div className="pipeline">
          <span className="badge accent">{STRATEGY_LABEL[recs.strategy] || recs.strategy}</span>
          {t.user_tower_ms != null && <span className="stage">User tower <strong>{ms(t.user_tower_ms)}</strong></span>}
          {t.retrieval_ms != null && <span className="stage">FAISS retrieval <strong>{ms(t.retrieval_ms)}</strong></span>}
          {t.ranking_ms != null && <span className="stage">LambdaRank <strong>{ms(t.ranking_ms)}</strong></span>}
          <span className="stage">Total <strong>{ms(t.total_ms)}</strong></span>
          <span className="muted small">model {recs.model_version}</span>
        </div>
      )}

      <ErrorBox error={error} />
      {loading && !recs ? <Spinner label="Ranking 25,612 products…" /> : recs && (
        <div className={`product-grid ${loading ? 'is-loading' : ''}`}>
          {recs.items.map((r) => (
            <ProductCard key={r.product.id} product={r.product} rank={r.position} onOpen={(p) => setOpen({ product: p, requestId: recs.request_id })}>
              <div className="reasons">
                {r.reasons.slice(0, 2).map((reason) => (
                  <span key={reason.feature} className="reason" title={reason.contribution != null ? `SHAP contribution ${reason.contribution.toFixed(3)}` : ''}>
                    {reason.label}
                  </span>
                ))}
              </div>
              {r.anchor && (
                <div className="anchor muted small" title={r.anchor.title}>
                  Because you liked <span className="anchor-title">{r.anchor.title}</span>
                </div>
              )}
              <div className="score-row muted small">
                <span>score {r.score.toFixed(3)}</span>
                {r.retrieval_rank != null && <span>retrieved #{r.retrieval_rank}</span>}
              </div>
            </ProductCard>
          ))}
        </div>
      )}

      {open && (
        <ProductDrawer
          key={open.product.id}
          product={open.product}
          requestId={open.requestId}
          onClose={() => setOpen(null)}
          onSelect={(p) => setOpen({ product: p })}
        />
      )}
    </div>
  )
}
