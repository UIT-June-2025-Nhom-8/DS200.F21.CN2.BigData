import { useAppStore } from '../../store/appStore'
import { VerdictBadge } from './VerdictBadge'
import { ConfidenceBar } from './ConfidenceBar'
import { MetricsTable } from './MetricsTable'

export function ResultsPanel() {
  const { appState, result, errorMsg } = useAppStore()

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl overflow-hidden">
      {/* Card header */}
      <div className="px-5 py-4 border-b border-[var(--border)] flex items-center justify-between">
        <div className="flex items-center gap-2 font-display text-[11px] font-semibold tracking-[0.12em] uppercase text-[var(--text-2)]">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--cyan)] animate-dot-pulse" style={{ boxShadow: '0 0 6px var(--cyan)' }} />
          Kết quả phân tích
        </div>
        <span className="font-mono text-[10px] text-[var(--text-3)]">EfficientNet-B0</span>
      </div>

      <div className="p-5">
        {/* Idle / analyzing */}
        {(appState === 'idle' || appState === 'analyzing') && (
          <div className="flex flex-col items-center justify-center min-h-[240px] gap-3 text-[var(--text-3)]">
            <div className="w-12 h-12 border border-[var(--border)] rounded-full flex items-center justify-center">
              <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 6v6l4 2" strokeLinecap="round" />
              </svg>
            </div>
            <p className="font-mono text-[11px] tracking-widest">
              {appState === 'analyzing' ? 'Đang phân tích...' : 'Đang chờ ảnh đầu vào...'}
            </p>
          </div>
        )}

        {/* Error */}
        {appState === 'error' && (
          <div className="flex flex-col items-center justify-center min-h-[240px] gap-3">
            <div
              className="w-12 h-12 rounded-full flex items-center justify-center"
              style={{ background: 'rgba(255,59,107,0.1)', border: '1px solid rgba(255,59,107,0.3)' }}
            >
              <svg width="22" height="22" fill="none" stroke="var(--red)" strokeWidth="1.5" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v4M12 16h.01" strokeLinecap="round" />
              </svg>
            </div>
            <p className="font-mono text-[11px] text-[var(--red)] tracking-wide text-center max-w-xs">
              {errorMsg ?? 'Lỗi không xác định.'}
            </p>
          </div>
        )}

        {/* Results */}
        {appState === 'results' && result && (
          <div className="animate-fade-in-up">
            <VerdictBadge result={result} />
            <ConfidenceBar label="Xác suất — Thật"     value={result.prob_real} color="green" />
            <ConfidenceBar label="Xác suất — Giả (AI)" value={result.prob_fake} color="red" />
            <div className="mt-5">
              <MetricsTable result={result} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
