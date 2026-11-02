import { useRef, useState } from 'react'

// Small, dependency-free charts. Marks carry color; text stays in text tokens.
// Every chart has a hover tooltip and a table view for accessibility.

function useTooltip() {
  const [tip, setTip] = useState(null)
  const show = (e, content) => setTip({ x: e.clientX, y: e.clientY, content })
  const hide = () => setTip(null)
  const node = tip && (
    <div className="chart-tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }} role="status">
      {tip.content}
    </div>
  )
  return { show, hide, node }
}

export function ChartFrame({ title, subtitle, table, children, legend }) {
  const [asTable, setAsTable] = useState(false)
  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <div>
          <div className="chart-title">{title}</div>
          {subtitle && <div className="chart-sub">{subtitle}</div>}
        </div>
        {table && (
          <button type="button" className="link-button small" onClick={() => setAsTable((v) => !v)}>
            {asTable ? 'Show chart' : 'Show table'}
          </button>
        )}
      </figcaption>
      {legend && !asTable && <div className="legend">{legend}</div>}
      {asTable ? table : children}
    </figure>
  )
}

export function DataTable({ columns, rows }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr>{columns.map((c) => <th key={c.key} className={c.num ? 'num' : ''}>{c.label}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{columns.map((c) => <td key={c.key} className={c.num ? 'num' : ''}>{c.format ? c.format(r[c.key], r) : r[c.key]}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** Horizontal bars; one series. `highlight` rows use the accent, others recede. */
export function HBarChart({ rows, format = (v) => v, max, tooltip }) {
  const tip = useTooltip()
  const top = max ?? (Math.max(...rows.map((r) => r.value), 0) * 1.08 || 1)
  return (
    <div className="hbar" onMouseLeave={tip.hide}>
      {rows.map((r) => (
        <div
          key={r.label}
          className="hbar-row"
          onMouseMove={(e) => tip.show(e, tooltip ? tooltip(r) : <><strong>{r.label}</strong><div>{format(r.value)}</div></>)}
        >
          <div className="hbar-label" title={r.label}>{r.label}</div>
          <div className="hbar-track">
            <div className={`hbar-fill ${r.highlight ? 'accent' : r.muted ? 'muted' : ''}`} style={{ width: `${(r.value / top) * 100}%` }} />
            <span className="hbar-value">{format(r.value)}</span>
          </div>
        </div>
      ))}
      {tip.node}
    </div>
  )
}

/** Two series per bucket (e.g. training vs live share). */
export function PairedBars({ rows, series, format = (v) => v }) {
  const tip = useTooltip()
  const top = Math.max(...rows.flatMap((r) => series.map((s) => r[s.key])), 0) * 1.08 || 1
  return (
    <div className="hbar" onMouseLeave={tip.hide}>
      {rows.map((r) => (
        <div
          key={r.label}
          className="hbar-row paired"
          onMouseMove={(e) => tip.show(e, (
            <>
              <strong>{r.label}</strong>
              {series.map((s) => (
                <div key={s.key} className="tip-row"><span className={`swatch ${s.className}`} />{s.label}: {format(r[s.key])}</div>
              ))}
            </>
          ))}
        >
          <div className="hbar-label" title={r.label}>{r.label}</div>
          <div className="hbar-stack">
            {series.map((s) => (
              <div key={s.key} className="hbar-track thin">
                <div className={`hbar-fill ${s.className}`} style={{ width: `${(r[s.key] / top) * 100}%` }} />
              </div>
            ))}
          </div>
        </div>
      ))}
      {tip.node}
    </div>
  )
}

/** Single-series line with crosshair tooltip. points: [{x, y}] */
export function LineChart({ points, format = (v) => v, xLabel = 'Epoch', height = 200, marker }) {
  const ref = useRef(null)
  const tip = useTooltip()
  const [hover, setHover] = useState(null)
  if (!points.length) return null
  const w = 600
  const pad = { l: 44, r: 16, t: 12, b: 28 }
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const x0 = Math.min(...xs)
  const x1 = Math.max(...xs)
  const yMax = Math.max(...ys) * 1.1
  const yMin = Math.max(0, Math.min(...ys) * 0.85)
  const sx = (x) => pad.l + ((x - x0) / Math.max(x1 - x0, 1)) * (w - pad.l - pad.r)
  const sy = (y) => pad.t + (1 - (y - yMin) / Math.max(yMax - yMin, 1e-9)) * (height - pad.t - pad.b)
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')
  const ticks = [0, 0.5, 1].map((t) => yMin + t * (yMax - yMin))

  const onMove = (e) => {
    const box = ref.current.getBoundingClientRect()
    const x = ((e.clientX - box.left) / box.width) * w
    let best = points[0]
    for (const p of points) if (Math.abs(sx(p.x) - x) < Math.abs(sx(best.x) - x)) best = p
    setHover(best)
    tip.show(e, <><strong>{xLabel} {best.x}</strong><div>{format(best.y)}</div></>)
  }

  return (
    <div className="line-chart">
      <svg ref={ref} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" onMouseMove={onMove}
        onMouseLeave={() => { setHover(null); tip.hide() }} role="img" aria-label="Line chart">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={pad.l - 6} y={sy(t) + 4} className="axis" textAnchor="end">{format(t)}</text>
          </g>
        ))}
        <text x={pad.l} y={height - 8} className="axis">{xLabel} {x0}</text>
        <text x={w - pad.r} y={height - 8} className="axis" textAnchor="end">{x1}</text>
        <path d={path} className="series-line" />
        {marker != null && (
          <circle cx={sx(marker.x)} cy={sy(marker.y)} r="5" className="series-dot" />
        )}
        {hover && (
          <>
            <line x1={sx(hover.x)} x2={sx(hover.x)} y1={pad.t} y2={height - pad.b} className="crosshair" />
            <circle cx={sx(hover.x)} cy={sy(hover.y)} r="5" className="series-dot" />
          </>
        )}
      </svg>
      {tip.node}
    </div>
  )
}

/** Latency meter against a budget. */
export function Meter({ value, budget, label }) {
  const ratio = Math.min(value / budget, 1)
  const tone = value <= budget * 0.5 ? 'good' : value <= budget ? 'warn' : 'critical'
  return (
    <div className="meter" aria-label={`${label}: ${value.toFixed(1)} of ${budget} ms budget`}>
      <div className="meter-track"><div className={`meter-fill ${tone}`} style={{ width: `${ratio * 100}%` }} /></div>
    </div>
  )
}
