export function ScanAnimation() {
  return (
    <div className="relative flex items-center justify-center w-[260px] h-[260px]">
      {/* Outer dashed ring */}
      <div
        className="absolute inset-[-40px] rounded-full border border-dashed border-[rgba(0,200,255,0.08)] animate-ring-spin-rev"
        style={{ borderStyle: 'dashed' }}
      />

      {/* Spinning ring with dot */}
      <div className="absolute inset-[-20px] rounded-full border border-[var(--border)] animate-ring-spin">
        <div
          className="absolute top-1/2 -left-1 w-2 h-2 rounded-full bg-[var(--cyan)] -translate-y-1/2"
          style={{ boxShadow: '0 0 10px var(--cyan)' }}
        />
      </div>

      {/* Face circle */}
      <div
        className="relative w-full h-full rounded-full border border-[var(--border)] overflow-hidden flex items-center justify-center"
        style={{ background: 'linear-gradient(135deg, var(--bg-card), var(--bg-hover))' }}
      >
        {/* Scan line */}
        <div className="absolute inset-4 rounded-full overflow-hidden pointer-events-none">
          <div
            className="absolute left-0 right-0 h-[2px] animate-scan-sweep"
            style={{
              background: 'linear-gradient(90deg, transparent, var(--cyan), transparent)',
              boxShadow: '0 0 14px var(--cyan)',
            }}
          />
        </div>

        {/* Face SVG */}
        <svg width="100" height="100" viewBox="0 0 120 120" fill="none" opacity={0.13}>
          <circle cx="60" cy="45" r="22" stroke="var(--cyan)" strokeWidth="1.5" />
          <path d="M20 100c5-18 18-28 40-28s35 10 40 28" stroke="var(--cyan)" strokeWidth="1.5" strokeLinecap="round" />
        </svg>

        {/* Blinking dots */}
        {[
          { top: '30%', left: '35%', delay: '0s' },
          { top: '40%', left: '60%', delay: '0.4s' },
          { top: '60%', left: '45%', delay: '0.8s' },
          { top: '50%', left: '30%', delay: '1.2s' },
          { top: '35%', left: '52%', delay: '0.6s' },
        ].map((pos, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 rounded-full bg-[var(--cyan)] animate-dot-pulse"
            style={{ top: pos.top, left: pos.left, animationDelay: pos.delay, boxShadow: '0 0 6px var(--cyan)' }}
          />
        ))}
      </div>

      {/* Label */}
      <div
        className="absolute -bottom-10 left-1/2 -translate-x-1/2 font-mono text-[10px] text-[var(--cyan-dim)] tracking-[0.15em] whitespace-nowrap animate-text-pulse"
      >
        SCANNING · REAL-TIME ANALYSIS
      </div>
    </div>
  )
}
