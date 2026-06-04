import type { PredictionResult } from '../../types/prediction'

export function VerdictBadge({ result }: { result: PredictionResult }) {
  const { is_fake, label, confidence } = result
  const color     = is_fake ? 'var(--red)'   : 'var(--green)'
  const dimColor  = is_fake ? 'rgba(255,59,107,0.06)' : 'rgba(0,255,157,0.04)'
  const borderClr = is_fake ? 'rgba(255,59,107,0.3)'  : 'rgba(0,255,157,0.3)'

  return (
    <div
      className="relative rounded-xl p-6 text-center overflow-hidden mb-5"
      style={{ background: `linear-gradient(135deg, var(--bg-surface), ${dimColor})`, border: `1px solid ${borderClr}` }}
    >
      {/* Shimmer */}
      <div
        className="absolute inset-0 pointer-events-none animate-shimmer"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.025), transparent)' }}
      />

      <p className="font-mono text-[10px] text-[var(--text-3)] tracking-[0.15em] uppercase mb-2">
        Phán quyết
      </p>
      <p
        className="font-display text-4xl font-bold tracking-widest leading-none mb-2"
        style={{ color, textShadow: `0 0 24px ${color}66` }}
      >
        {label}
      </p>
      <p className="font-mono text-[13px] text-[var(--text-2)]">
        Độ tin cậy: <strong className="text-[var(--text-1)]">{(confidence * 100).toFixed(1)}%</strong>
      </p>
    </div>
  )
}
