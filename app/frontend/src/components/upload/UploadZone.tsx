import { useRef, useState, type DragEvent, type ChangeEvent } from 'react'
import { useAppStore } from '../../store/appStore'

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']
const MAX_BYTES = 10 * 1024 * 1024

interface Props {
  onToast: (msg: string) => void
}

export function UploadZone({ onToast }: Props) {
  const { setFile, analyze } = useAppStore()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function handleFile(file: File) {
    if (!ACCEPTED.includes(file.type)) {
      onToast('Chỉ chấp nhận JPG, PNG, WEBP.')
      return
    }
    if (file.size > MAX_BYTES) {
      onToast('File vượt quá 10MB.')
      return
    }
    setFile(file)
    analyze()
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''
  }

  return (
    <div
      className={`relative rounded-2xl border-2 border-dashed px-8 py-16 text-center cursor-pointer
        transition-all duration-200 overflow-hidden
        ${dragging
          ? 'border-[var(--cyan-dim)] bg-[var(--bg-card)]'
          : 'border-[var(--border)] bg-[var(--bg-surface)] hover:border-[var(--cyan-dim)] hover:bg-[var(--bg-card)]'
        }`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      {/* Radial glow on hover */}
      <div
        className={`absolute inset-0 pointer-events-none transition-opacity duration-200
          bg-[radial-gradient(ellipse_at_center,rgba(0,200,255,0.04)_0%,transparent_70%)]
          ${dragging ? 'opacity-100' : 'opacity-0'}`}
      />

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        className="hidden"
        onChange={onChange}
      />

      {/* Upload icon */}
      <div
        className={`w-14 h-14 mx-auto mb-5 rounded-full border flex items-center justify-center
          transition-all duration-200
          ${dragging
            ? 'border-[var(--cyan)] text-[var(--cyan)] shadow-[0_0_20px_rgba(0,200,255,0.2)]'
            : 'border-[var(--border)] text-[var(--cyan-dim)]'
          }`}
      >
        <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      <p className="font-display text-[15px] font-semibold text-[var(--text-1)] mb-2">
        Kéo thả ảnh khuôn mặt vào đây
      </p>
      <p className="text-[var(--text-3)] text-[13px] mb-6">hoặc nhấn để chọn file từ máy tính</p>

      <button
        className="font-display text-[13px] font-semibold tracking-wider text-[var(--bg-base)]
          bg-[var(--cyan)] px-6 py-2.5 rounded-lg border-none cursor-pointer
          hover:bg-[#33d6ff] hover:shadow-[0_0_24px_rgba(0,200,255,0.4)]
          active:scale-[0.98] transition-all duration-200"
        onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}
      >
        Chọn ảnh
      </button>

      <p className="font-mono text-[10px] text-[var(--text-3)] tracking-widest mt-4">
        JPG · PNG · WEBP · tối đa 10MB
      </p>
    </div>
  )
}
