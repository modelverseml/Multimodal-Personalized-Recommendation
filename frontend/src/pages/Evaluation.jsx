import { useEffect, useState } from 'react'
import { api } from '../api'
import { ChartFrame, DataTable, HBarChart, LineChart, Meter } from '../components/charts'
import { Empty, ErrorBox, Spinner, Stat } from '../components/ui'
import { ms, pct } from '../format'

const OURS = new Set(['Two-Tower (multimodal)', 'Two-Tower + LambdaRank'])
const VARIANT_LABEL = {
  id_only: 'ID only (≈ SASRec)',
  id_text: 'ID + CLIP text',
  content_only: 'CLIP text + image, no ID',
  full: 'ID + CLIP text + image',
}
const fmt4 = (v) => v.toFixed(4)

export default function Evaluation() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [metric, setMetric] = useState('ndcg@10')

  useEffect(() => { api.evaluation().then(setData).catch(setError) }, [])

  if (error) return <div className="page"><ErrorBox error={error} /></div>
  if (!data) return <div className="page"><Spinner /></div>
  const { report, training } = data
  if (!report) {
    return (
      <div className="page">
        <div className="page-head"><h1>Evaluation</h1></div>
        <Empty title="No evaluation report yet">Run <code>python -m scripts.evaluate</code> in the backend.</Empty>
      </div>
    )
  }

  const results = report.results
  const ours = results.find((r) => r.model === 'Two-Tower + LambdaRank')
  const lift = report.lift_vs_best_baseline
  const e2e = report.latency['with_ranker.total_ms']
  const budget = 100

  const modelRows = [...results].sort((a, b) => a[metric] - b[metric]).map((r) => ({
    label: r.model, value: r[metric], highlight: OURS.has(r.model), muted: !OURS.has(r.model), row: r,
  }))
  const ablationRows = report.ablations
    .map((a) => ({ label: VARIANT_LABEL[a.variant] || a.variant, value: a['ndcg@10'], highlight: a.variant === 'full', muted: a.variant !== 'full' }))
    .sort((a, b) => a.value - b.value)
  const importance = training.ranker?.feature_importance || {}
  const totalGain = Object.values(importance).reduce((a, b) => a + b, 0) || 1
  const importanceRows = Object.entries(importance)
    .map(([k, v]) => ({ label: k.replaceAll('_', ' '), value: v / totalGain }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10)
  const curve = training.history.map((h) => ({ x: h.epoch, y: h['ndcg@10'] }))
  const best = training.history.reduce((a, b) => (b['ndcg@10'] > a['ndcg@10'] ? b : a), training.history[0] || {})
  const stages = [
    ['User tower', report.latency['with_ranker.user_tower_ms']],
    ['FAISS retrieval', report.latency['with_ranker.retrieval_ms']],
    ['LambdaRank + features', report.latency['with_ranker.ranking_ms']],
    ['End to end', e2e],
  ].filter(([, v]) => v)

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Offline evaluation</h1>
          <p className="muted">
            {report.dataset}, {report.n_users.toLocaleString()} test users, full ranking over all {report.n_items.toLocaleString()} items
            ({report.protocol}). Model <code>{report.model_version}</code>, MLflow run <code>{report.run_id.slice(0, 12)}</code>.
          </p>
        </div>
      </div>

      <div className="stats">
        <Stat label="NDCG@10" value={ours['ndcg@10'].toFixed(4)}
          hint={`${lift['ndcg@10_relative_lift'] >= 0 ? '+' : ''}${pct(lift['ndcg@10_relative_lift'])} vs ${lift.baseline}`} />
        <Stat label="Recall@10" value={ours['recall@10'].toFixed(4)}
          hint={`${lift['recall@10_relative_lift'] >= 0 ? '+' : ''}${pct(lift['recall@10_relative_lift'])} vs ${lift.baseline}`} />
        <Stat label="Candidate recall@100" value={pct(report['candidate_recall@100'])} hint="Upper bound for the reranker" />
        <Stat label="p95 serving latency" value={ms(e2e.p95_ms)} hint={`p99 ${ms(e2e.p99_ms)} · budget ${budget} ms`}
          tone={e2e.p95_ms <= budget ? 'good' : 'critical'} />
      </div>

      <div className="grid-2">
        <section className="card">
          <ChartFrame
            title="Model comparison"
            subtitle={`Test ${metric.toUpperCase()}, bootstrap 95% CI in tooltip. Blue = this project.`}
            table={<DataTable rows={results} columns={[
              { key: 'model', label: 'Model' },
              { key: 'recall@10', label: 'Recall@10', num: true, format: fmt4 },
              { key: 'ndcg@10', label: 'NDCG@10', num: true, format: fmt4 },
              { key: 'recall@20', label: 'Recall@20', num: true, format: fmt4 },
              { key: 'mrr', label: 'MRR', num: true, format: fmt4 },
              { key: 'coverage@10', label: 'Coverage@10', num: true, format: (v) => pct(v) },
            ]} />}
          >
            <div className="row gap chart-controls">
              {['ndcg@10', 'recall@10', 'recall@20', 'mrr'].map((m) => (
                <button type="button" key={m} className={`chip ${metric === m ? 'active' : ''}`} onClick={() => setMetric(m)}>{m.toUpperCase()}</button>
              ))}
            </div>
            <HBarChart rows={modelRows} format={fmt4} tooltip={(r) => (
              <>
                <strong>{r.label}</strong>
                <div>{metric.toUpperCase()}: {fmt4(r.value)}</div>
                {metric === 'ndcg@10' && <div className="muted">95% CI {r.row['ndcg@10_ci95'].map(fmt4).join(' – ')}</div>}
                <div className="muted">Coverage@10 {pct(r.row['coverage@10'])} · diversity {r.row['category_diversity@10'].toFixed(2)}</div>
              </>
            )} />
          </ChartFrame>
        </section>

        <section className="card">
          <ChartFrame
            title="Serving latency"
            subtitle={`${report.latency['with_ranker.total_ms'] ? 'In-process CPU, per request' : ''}; budget ${budget} ms`}
            table={<DataTable rows={stages.map(([label, v]) => ({ label, ...v }))} columns={[
              { key: 'label', label: 'Stage' },
              { key: 'p50_ms', label: 'p50', num: true, format: ms },
              { key: 'p95_ms', label: 'p95', num: true, format: ms },
              { key: 'p99_ms', label: 'p99', num: true, format: ms },
            ]} />}
          >
            <div className="latency-list">
              {stages.map(([label, v]) => (
                <div key={label} className="latency-row">
                  <div className="row between"><span>{label}</span><span className="muted small">p50 {ms(v.p50_ms)} · <strong>p95 {ms(v.p95_ms)}</strong></span></div>
                  <Meter value={v.p95_ms} budget={budget} label={label} />
                </div>
              ))}
            </div>
            <div className="faiss-note">
              <div className="row between"><span>FAISS HNSW recall@{report.faiss.k} vs exact search</span><strong>{pct(report.faiss.hnsw_recall_vs_exact)}</strong></div>
              <div className="row between muted small">
                <span>HNSW p50 {ms(report.faiss.hnsw.p50_ms)} vs exact {ms(report.faiss.exact_flat.p50_ms)}</span>
                <span>M={report.faiss.hnsw_params.M}, efSearch={report.faiss.hnsw_params.efSearch}</span>
              </div>
              {report.semantic_search && (
                <div className="row between muted small">
                  <span>CLIP semantic search (ONNX text encoder + FAISS)</span>
                  <span>p50 {ms(report.semantic_search.p50_ms)} · p95 {ms(report.semantic_search.p95_ms)}</span>
                </div>
              )}
            </div>
          </ChartFrame>
        </section>

        <section className="card">
          <ChartFrame
            title="Ablation: what each modality adds"
            subtitle="Two-tower retrieval only, test NDCG@10"
            table={<DataTable rows={report.ablations} columns={[
              { key: 'variant', label: 'Variant', format: (v) => VARIANT_LABEL[v] || v },
              { key: 'recall@10', label: 'Recall@10', num: true, format: fmt4 },
              { key: 'ndcg@10', label: 'NDCG@10', num: true, format: fmt4 },
              { key: 'best_epoch', label: 'Best epoch', num: true },
            ]} />}
          >
            <HBarChart rows={ablationRows} format={fmt4} />
          </ChartFrame>
        </section>

        <section className="card">
          <ChartFrame
            title="Training curve"
            subtitle={`Validation NDCG@10 per epoch; best at epoch ${best.epoch} (early stopping)`}
            table={<DataTable rows={training.history} columns={[
              { key: 'epoch', label: 'Epoch', num: true },
              { key: 'loss', label: 'Loss', num: true, format: (v) => v.toFixed(3) },
              { key: 'recall@10', label: 'Val recall@10', num: true, format: fmt4 },
              { key: 'ndcg@10', label: 'Val NDCG@10', num: true, format: fmt4 },
            ]} />}
          >
            <LineChart points={curve} format={(v) => v.toFixed(3)} marker={{ x: best.epoch, y: best['ndcg@10'] }} />
          </ChartFrame>
        </section>

        {importanceRows.length > 0 && (
          <section className="card span-2">
            <ChartFrame
              title="Reranker feature importance"
              subtitle="Share of total LightGBM split gain"
              table={<DataTable rows={importanceRows} columns={[
                { key: 'label', label: 'Feature' },
                { key: 'value', label: 'Gain share', num: true, format: (v) => pct(v) },
              ]} />}
            >
              <HBarChart rows={importanceRows} format={(v) => pct(v)} />
            </ChartFrame>
          </section>
        )}
      </div>
    </div>
  )
}
