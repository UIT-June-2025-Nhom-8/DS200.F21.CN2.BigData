export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[rgba(5,8,15,0.85)] backdrop-blur-xl">
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="15" stroke="var(--cyan)" strokeWidth="1" />
            <circle cx="16" cy="16" r="10" stroke="var(--cyan)" strokeWidth="0.5" strokeDasharray="2 2" />
            <circle cx="16" cy="12" r="4" stroke="var(--cyan)" strokeWidth="1" />
            <path d="M8 24c1.5-3 4.5-5 8-5s6.5 2 8 5" stroke="var(--cyan)" strokeWidth="1" strokeLinecap="round" />
          </svg>
          <span className="font-display text-[17px] font-bold tracking-wide">
            Face<span className="text-[var(--cyan)]">Guard</span>
          </span>
          <span className="font-mono text-[10px] text-[var(--cyan-dim)] border border-[var(--border)] px-2.5 py-1 rounded-full tracking-widest hidden sm:inline">
            EfficientNet-B0 · v1.0
          </span>
        </div>

        {/* Live metrics */}
        <div className="hidden sm:flex gap-6">
          {[
            { val: '97.33%', label: 'Accuracy' },
            { val: '99.70%', label: 'AUC-ROC' },
          ].map(({ val, label }) => (
            <div key={label} className="text-right">
              <div className="font-display text-[15px] font-semibold text-[var(--cyan)] leading-none">{val}</div>
              <div className="font-mono text-[9px] text-[var(--text-3)] tracking-widest uppercase mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      </div>
    </header>
  )
}
