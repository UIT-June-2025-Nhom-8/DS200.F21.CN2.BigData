export function Footer() {
  return (
    <footer className="border-t border-[var(--border)] mt-24">
      <div className="max-w-6xl mx-auto px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-2">
        <p className="font-mono text-[11px] text-[var(--text-3)]">
          <span className="text-[var(--cyan-dim)]">FaceGuard</span> · DS200.F21.CN2 · UIT-VNU HCM · 2025
        </p>
        <p className="font-mono text-[11px] text-[var(--text-3)]">
          EfficientNet-B0 · Stage 3 · AUC-ROC{' '}
          <span className="text-[var(--cyan-dim)]">99.70%</span>
        </p>
      </div>
    </footer>
  )
}
