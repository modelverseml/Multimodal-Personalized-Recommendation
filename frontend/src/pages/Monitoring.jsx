import { useState } from 'react'
import { api } from '../api'
import { ChartFrame, DataTable, PairedBars } from '../components/charts'
import { RefreshIcon } from '../components/icons'
import { ErrorBox, Segmented, Spinner, Stat } from '../components/ui'
import { pct } from '../format'
import { useAsync } from '../useAsync'

const STATUS = {
  stable: { label: 'Stable', tone: 'good', icon: '●' },
  warning: { label: 'Watch', tone: 'warn', icon: '▲' },
  drift: { label: 'Drift detected', tone: 'critical', icon: '■' },
  insufficient_data: { label: 'Not enough live data', tone: null, icon: '○' },
}

export default function Monitoring() {
  const [days, setDays] = useState(7)
  const { data, error, reload: load } = useAsync(() => api.monitoring(days), [days])

  if (error && !data) return <div className="page"><ErrorBox error={error} /></div>
  if (!data) return <div className="page"><Spinner /></div>
  const { drift, engagement } = data
  const status = STATUS[drift.status]
  const rows = drift.buckets.map((b) => ({ label: b.bucket, expected: b.expected, actual: b.actual, psi: b.psi }))

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Monitoring</h1>
          <p className="muted">
            Population Stability Index between the training category mix and live traffic, plus engagement
            on logged recommendation impressions. The PSI ≥ {drift.threshold} cutoff is a monitoring heuristic, not an automatic retraining rule.
          </p>
        </div>
        <div className="row gap">
          <Segmented label="Window" value={days} onChange={setDays}
            options={[{ value: 1, label: '24h' }, { value: 7, label: '7 days' }, { value: 30, label: '30 days' }]} />
          <button type="button" className="icon-button" onClick={load} aria-label="Refresh"><RefreshIcon /></button>
        </div>
      </div>

      <div className="stats">
        <Stat label="Drift status" value={<span><span aria-hidden="true">{status.icon}</span> {status.label}</span>} tone={status.tone}
          hint={drift.psi != null ? `PSI ${drift.psi.toFixed(3)}` : drift.message} />
        <Stat label="Live events" value={drift.live_events.toLocaleString()} hint={`last ${days} day${days > 1 ? 's' : ''}`} />
        <Stat label="Recommendation CTR" value={pct(engagement.ctr)} hint={`${engagement.impressions.toLocaleString()} impressions, ${engagement.requests} requests`} />
        <Stat label="Conversion per impression" value={pct(engagement.conversion_rate, 2)}
          hint={`${engagement.attributed_events.purchase || 0} attributed purchases`} />
      </div>

      <section className="card">
        {rows.length === 0 ? (
          <div className="empty">
            <div className="empty-title">No live traffic to compare yet</div>
            <p className="muted">
              Browse as a few shoppers, or run <code>python -m scripts.simulate --shoppers 40 --platform "Nintendo Switch"</code> to
              generate skewed traffic and watch PSI respond.
            </p>
          </div>
        ) : (
          <ChartFrame
            title="Category mix: training vs live"
            subtitle="Largest PSI contributors first"
            legend={<>
              <span className="legend-item"><span className="swatch series-1" />Training share</span>
              <span className="legend-item"><span className="swatch series-2" />Live share</span>
            </>}
            table={<DataTable rows={rows} columns={[
              { key: 'label', label: 'Category' },
              { key: 'expected', label: 'Training', num: true, format: (v) => pct(v) },
              { key: 'actual', label: 'Live', num: true, format: (v) => pct(v) },
              { key: 'psi', label: 'PSI term', num: true, format: (v) => v.toFixed(4) },
            ]} />}
          >
            <PairedBars
              rows={rows}
              format={(v) => pct(v)}
              series={[
                { key: 'expected', label: 'Training', className: 'series-1' },
                { key: 'actual', label: 'Live', className: 'series-2' },
              ]}
            />
          </ChartFrame>
        )}
      </section>
    </div>
  )
}
