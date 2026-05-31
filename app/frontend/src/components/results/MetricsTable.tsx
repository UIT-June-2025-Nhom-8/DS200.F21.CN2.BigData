import type { PredictionResult } from '../../types/prediction'

export function MetricsTable({ result }: { result: PredictionResult }) {
  const rows = [
    { name: 'Logit score',      val: result.logit.toFixed(3) },
    { name: 'Inference time',   val: `${result.inference_ms} ms` },
    { name: 'Model',            val: 'EfficientNet-B0' },
    { name: 'Stage checkpoint', val: 'Stage 3' },
  ]

  return (
    <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      {rows.map((row, i) => (
        <div
          key={row.name}
          className="flex justify-between items-center px-4 py-2.5 transition-colors duration-150
            hover:bg-[var(--bg-hover)]"
          style={{ background: 'var(--bg-surface)', borderTop: i > 0 ? '1px solid var(--border)' : 'none' }}
        >
          <span className="font-mono text-[11px] text-[var(--text-3)] tracking-wide">{row.name}</span>
          <span className="font-display text-[12px] font-semibold text-[var(--cyan)]">{row.val}</span>
        </div>
      ))}
    </div>
  )
}
