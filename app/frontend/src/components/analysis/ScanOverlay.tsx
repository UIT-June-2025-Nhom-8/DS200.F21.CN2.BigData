export function ScanOverlay() {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center bg-[rgba(5,8,15,0.88)] z-10">
      {/* Ring progress */}
      <div className="w-20 h-20 mb-4">
        <svg width="80" height="80" viewBox="0 0 80 80" style={{ transform: 'rotate(-90deg)' }}>
          <circle cx="40" cy="40" r="35" fill="none" stroke="var(--border)" strokeWidth="2" />
          <circle
            cx="40" cy="40" r="35"
            fill="none"
            stroke="var(--cyan)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeDasharray="220"
            strokeDashoffset="220"
            className="animate-ring-progress"
            style={{ filter: 'drop-shadow(0 0 4px var(--cyan))' }}
          />
        </svg>
      </div>
      <p className="font-mono text-[11px] text-[var(--cyan)] tracking-[0.15em] animate-text-pulse">
        ANALYZING IMAGE...
      </p>
    </div>
  )
}
