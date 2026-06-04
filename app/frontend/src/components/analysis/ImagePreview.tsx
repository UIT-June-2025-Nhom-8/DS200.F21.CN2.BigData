import { useAppStore } from '../../store/appStore'
import { ScanOverlay } from './ScanOverlay'
import { Button } from '../ui/Button'

interface Props {
  onToast: (msg: string) => void
  onNewFile: () => void
}

export function ImagePreview({ onToast, onNewFile }: Props) {
  const { appState, previewUrl, result, gradcamVisible, toggleGradcam, reset } = useAppStore()

  const isAnalyzing = appState === 'analyzing'
  const hasResult   = appState === 'results' || appState === 'error'

  const gradcamSrc = result?.gradcam_b64
    ? `data:image/png;base64,${result.gradcam_b64}`
    : null

  function handleToggleGradcam() {
    if (!gradcamSrc) {
      onToast('Grad-CAM không khả dụng cho ảnh này.')
      return
    }
    toggleGradcam()
    onToast(gradcamVisible ? 'Grad-CAM: TẮT' : 'Grad-CAM: BẬT')
  }

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl overflow-hidden">
      {/* Card header */}
      <div className="px-5 py-4 border-b border-[var(--border)] flex items-center justify-between">
        <div className="flex items-center gap-2 font-display text-[11px] font-semibold tracking-[0.12em] uppercase text-[var(--text-2)]">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--cyan)] animate-dot-pulse" style={{ boxShadow: '0 0 6px var(--cyan)' }} />
          Ảnh đầu vào
        </div>
      </div>

      {/* Image area */}
      <div className="relative min-h-[300px] bg-[var(--bg-surface)] flex items-center justify-center">
        {previewUrl ? (
          /* Wrap ảnh + overlay cùng container — overlay theo đúng kích thước ảnh gốc */
          <div className="relative p-4 inline-flex">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-w-full max-h-72 rounded-lg block"
            />
            {gradcamSrc && (
              <img
                src={gradcamSrc}
                alt="Grad-CAM"
                className="absolute top-4 left-4 rounded-lg"
                style={{
                  width:      'calc(100% - 32px)',
                  height:     'calc(100% - 32px)',
                  objectFit:  'fill',          /* 224×224 → khớp pixel-perfect */
                  opacity:    gradcamVisible ? 1 : 0,
                  transition: 'opacity 0.35s ease',
                }}
              />
            )}
          </div>
        ) : (
          <div className="text-center text-[var(--text-3)]">
            <svg className="w-12 h-12 mx-auto mb-3 opacity-20" fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 48 48">
              <rect x="4" y="8" width="40" height="32" rx="3" />
              <circle cx="16" cy="20" r="4" />
              <path d="M4 32l10-8 8 6 8-10 14 12" />
            </svg>
            <p className="font-mono text-[11px] tracking-widest">Chưa có ảnh</p>
          </div>
        )}

        {/* Scan overlay */}
        {isAnalyzing && <ScanOverlay />}
      </div>

      {/* Controls */}
      {hasResult && (
        <div className="px-5 py-3 border-t border-[var(--border)] flex gap-2">
          <Button
            variant="ghost"
            className={`flex-1 ${gradcamVisible ? '!border-[var(--cyan)] !text-[var(--cyan)] !bg-[rgba(0,200,255,0.1)]' : ''}`}
            onClick={handleToggleGradcam}
            disabled={!gradcamSrc}
          >
            Grad-CAM
          </Button>
          <Button variant="ghost" className="flex-1" onClick={reset}>
            Đặt lại
          </Button>
          <Button variant="primary" className="flex-1" onClick={onNewFile}>
            Ảnh mới
          </Button>
        </div>
      )}
    </div>
  )
}
