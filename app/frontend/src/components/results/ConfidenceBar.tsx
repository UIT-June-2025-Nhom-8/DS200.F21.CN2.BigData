import { useEffect, useRef } from 'react'

interface Props {
  label: string
  value: number   // 0–1
  color: 'green' | 'red'
}

const colors = {
  green: { fill: 'linear-gradient(90deg, var(--green), #00ffcc)', glow: 'var(--green)' },
  red:   { fill: 'linear-gradient(90deg, var(--red), #ff6b8a)',   glow: 'var(--red)'   },
}

export function ConfidenceBar({ label, value, color }: Props) {
  const barRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const bar = barRef.current
    if (!bar) return
    bar.style.width = '0%'
    const raf = requestAnimationFrame(() => {
      bar.style.transition = 'width 1s cubic-bezier(0.4,0,0.2,1)'
      bar.style.width = `${value * 100}%`
    })
    return () => cancelAnimationFrame(raf)
  }, [value])

  const { fill, glow } = colors[color]

  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-1.5">
        <span className="font-mono text-[11px] text-[var(--text-3)] tracking-widest">{label}</span>
        <span className="font-display text-[13px] font-bold text-[var(--text-1)]">
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-1 bg-[var(--bg-surface)] rounded-full overflow-hidden">
        <div
          ref={barRef}
          className="h-full rounded-full"
          style={{ background: fill, boxShadow: `0 0 8px ${glow}` }}
        />
      </div>
    </div>
  )
}
