import { useEffect, useState } from 'react'
import { api, prefs } from '../api'
import { SearchIcon } from '../components/icons'
import ProductCard from '../components/ProductCard'
import ProductDrawer from '../components/ProductDrawer'
import { Empty, ErrorBox, Segmented, Spinner } from '../components/ui'
import { compact, ms } from '../format'
import { useAsync } from '../useAsync'

const SEMANTIC_EXAMPLES = ['cute colorful platformer for kids', 'dark horror survival game', 'racing wheel controller', 'retro pixel art RPG']

export default function Shop() {
  const [mode, setMode] = useState(() => prefs.get('searchMode', 'semantic'))
  const [query, setQuery] = useState('')
  const [submitted, setSubmitted] = useState('')
  const [category, setCategory] = useState('')
  const [sort, setSort] = useState('popular')
  const [page, setPage] = useState(1)
  const [open, setOpen] = useState(null)
  const { data: categories = [] } = useAsync(() => api.categories(), [])
  useEffect(() => { prefs.set('searchMode', mode) }, [mode])

  const semantic = mode === 'semantic' && Boolean(submitted)
  const { data, error, loading } = useAsync(() => (semantic
    ? api.search(submitted, 48).then((r) => ({ items: r.items.map((i) => i.product), total: r.items.length, latency: r.latency_ms, semantic: true }))
    : api.products({ q: submitted || undefined, category: category || undefined, sort, page, page_size: 24 })
  ), [semantic, submitted, category, sort, page])

  const submit = (e) => {
    e.preventDefault()
    setPage(1)
    setSubmitted(query.trim())
  }

  const pages = data && !data.semantic ? Math.ceil(data.total / 24) : 1

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Catalog</h1>
          <p className="muted">
            {compact(25612)} video-game products. Semantic search embeds your query with the CLIP text encoder and
            matches it against product <em>images</em> and descriptions in the same embedding space.
          </p>
        </div>
      </div>

      <form className="search-bar" onSubmit={submit}>
        <SearchIcon />
        <input
          className="search-input"
          placeholder={mode === 'semantic' ? 'Describe what you want, e.g. "cozy farming game"' : 'Search titles and brands'}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search"
        />
        <Segmented
          label="Search mode"
          value={mode}
          onChange={(m) => { setMode(m); setPage(1) }}
          options={[{ value: 'semantic', label: 'Semantic (CLIP)' }, { value: 'keyword', label: 'Keyword' }]}
        />
        <button className="button primary">Search</button>
      </form>

      {mode === 'semantic' && !submitted && (
        <div className="examples">
          <span className="muted small">Try:</span>
          {SEMANTIC_EXAMPLES.map((ex) => (
            <button type="button" key={ex} className="chip" onClick={() => { setQuery(ex); setSubmitted(ex) }}>{ex}</button>
          ))}
        </div>
      )}

      {!(mode === 'semantic' && submitted) && (
        <div className="filters">
          <select className="input select" value={category} onChange={(e) => { setCategory(e.target.value); setPage(1) }} aria-label="Platform">
            <option value="">All platforms</option>
            {categories.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.count})</option>)}
          </select>
          <select className="input select" value={sort} onChange={(e) => { setSort(e.target.value); setPage(1) }} aria-label="Sort">
            <option value="popular">Most reviewed</option>
            <option value="rating">Top rated</option>
            <option value="price_asc">Price: low to high</option>
            <option value="price_desc">Price: high to low</option>
          </select>
          {data && <span className="muted small">{data.total.toLocaleString()} products</span>}
        </div>
      )}
      {data?.semantic && (
        <div className="filters">
          <span className="badge accent">CLIP text → image + text</span>
          <span className="muted small">Top {data.items.length} matches for “{submitted}” · query encoded and searched in {ms(data.latency)}</span>
          <button type="button" className="link-button small" onClick={() => { setQuery(''); setSubmitted('') }}>Clear</button>
        </div>
      )}

      <ErrorBox error={error} />
      {loading && !data ? <Spinner /> : data && data.items.length === 0 ? (
        <Empty title="No products found">Try a different search or platform.</Empty>
      ) : data && (
        <div className={`product-grid ${loading ? 'is-loading' : ''}`}>
          {data.items.map((p) => <ProductCard key={p.id} product={p} onOpen={(prod) => setOpen(prod)} />)}
        </div>
      )}

      {pages > 1 && (
        <div className="pager">
          <button type="button" className="button" disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous</button>
          <span className="muted">Page {page} of {pages.toLocaleString()}</span>
          <button type="button" className="button" disabled={page >= pages} onClick={() => setPage(page + 1)}>Next</button>
        </div>
      )}

      {open && <ProductDrawer key={open.id} product={open} onClose={() => setOpen(null)} onSelect={setOpen} />}
    </div>
  )
}
