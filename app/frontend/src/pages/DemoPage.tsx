import { useRef, useState, useCallback, type ChangeEvent } from 'react'
import { HeroSection } from '../components/hero/HeroSection'
import { UploadZone } from '../components/upload/UploadZone'
import { ImagePreview } from '../components/analysis/ImagePreview'
import { ResultsPanel } from '../components/results/ResultsPanel'
import { Toast } from '../components/ui/Toast'
import { useAppStore } from '../store/appStore'

const features = [
  {
    title: 'Grad-CAM Visualization',
    desc: 'Hiển thị vùng khuôn mặt mà mô hình tập trung phân tích, giúp giải thích kết quả phán đoán một cách trực quan.',
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: 'Phân tích tức thì',
    desc: 'Kết quả trả về dưới 1 giây với EfficientNet-B0 tối ưu, hỗ trợ cả CPU, GPU và Apple Silicon (MPS).',
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: 'Huấn luyện 3 giai đoạn',
    desc: 'Transfer learning theo giai đoạn: đông đặc backbone → mở từng lớp → fine-tune toàn bộ, đạt AUC-ROC 99.70%.',
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: 'Cross-generator Testing',
    desc: 'Đánh giá khả năng tổng quát trên dữ liệu từ StyleGAN, Deepfake, ciplab — không bị overfit một nguồn.',
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7C5 4 4 5 4 7z" />
        <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: 'Robustness Testing',
    desc: 'Đánh giá độ bền với nén JPEG (q=30–70), giảm kích thước và Gaussian blur để đảm bảo hiệu suất thực tế.',
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M12 15a3 3 0 100-6 3 3 0 000 6z" />
        <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: '280k+ Training Images',
    desc: 'Kết hợp 4 bộ dữ liệu: 140k-StyleGAN, Deepfake-Real, Hard-FakeReal, ciplab — đa dạng và cân bằng lớp.',
    icon: (
      <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
        <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
]

const members = [
  { mssv: '25410056', name: 'Lã Xuân Hồng' },
  { mssv: '25410150', name: 'Nguyễn Minh Trọng' },
  { mssv: '25410034', name: 'Lê Quang Hoài Đức' },
  { mssv: '25410104', name: 'Nguyễn Minh Nhật' },
  { mssv: '25410088', name: 'Trần Thanh Long' },
]

const archSteps = [
  { num: '01', name: 'Ảnh đầu vào',    detail: 'Resize 224×224\nImageNet normalize' },
  { num: '02', name: 'EfficientNet-B0', detail: 'Backbone frozen\nFeature extraction' },
  { num: '03', name: 'Custom Head',     detail: 'Linear(1280→256)\nReLU · Dropout(0.5)' },
  { num: '04', name: 'Classifier',      detail: 'Linear(256→1)\nSigmoid → Score' },
  { num: '05', name: 'Kết quả',         detail: 'THẬT / GIẢ\n+ Grad-CAM map' },
]

export function DemoPage() {
  const { appState, setFile, analyze } = useAppStore()
  const previewUrl = useAppStore((s) => s.previewUrl)
  const [toast, setToast] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleToast = useCallback((msg: string) => setToast(msg), [])
  const handleNewFile = useCallback(() => fileInputRef.current?.click(), [])

  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setToast('Chỉ chấp nhận JPG, PNG, WEBP.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setToast('File vượt quá 10MB.')
      return
    }
    setFile(file)
    analyze()
    e.target.value = ''
  }, [setFile, analyze])

  const showAppGrid = appState !== 'idle' || previewUrl !== null

  return (
    <>
      {/* Hero */}
      <HeroSection />

      {/* Divider */}
      <div className="max-w-6xl mx-auto px-6">
        <div className="h-px" style={{ background: 'linear-gradient(90deg, transparent, var(--border), transparent)' }} />
      </div>

      {/* Demo section */}
      <section className="max-w-6xl mx-auto px-6 py-12">
        {/* Upload zone — shown when idle with no file */}
        {!showAppGrid && (
          <div className="animate-fade-in-up">
            <UploadZone onToast={handleToast} />
          </div>
        )}

        {/* App grid — shown once file is selected */}
        {showAppGrid && (
          <div
            className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in-up"
          >
            <ImagePreview onToast={handleToast} onNewFile={handleNewFile} />
            <ResultsPanel />
          </div>
        )}
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 py-16 border-t border-[var(--border)]">
        <div className="flex items-center gap-2 font-mono text-[11px] text-[var(--cyan)] tracking-[0.15em] uppercase mb-3">
          <span className="w-4 h-px bg-[var(--cyan)]" />
          Tính năng
        </div>
        <h2 className="font-display text-2xl lg:text-3xl font-bold mb-10">Được thiết kế cho độ tin cậy</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((f) => (
            <div
              key={f.title}
              className="relative bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl p-7
                hover:border-[var(--border-glow)] hover:bg-[var(--bg-card)] hover:-translate-y-0.5
                transition-all duration-200 overflow-hidden group"
            >
              <div className="absolute top-0 left-0 right-0 h-[2px] opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                style={{ background: 'linear-gradient(90deg, transparent, var(--cyan), transparent)' }} />
              <div className="w-10 h-10 border border-[var(--border)] rounded-lg flex items-center justify-center text-[var(--cyan)] mb-4">
                {f.icon}
              </div>
              <h3 className="font-display text-[14px] font-semibold mb-2">{f.title}</h3>
              <p className="text-[13px] text-[var(--text-3)] leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Architecture pipeline */}
      <section className="max-w-6xl mx-auto px-6 py-16 border-t border-[var(--border)]">
        <div className="flex items-center gap-2 font-mono text-[11px] text-[var(--cyan)] tracking-[0.15em] uppercase mb-3">
          <span className="w-4 h-px bg-[var(--cyan)]" />
          Pipeline
        </div>
        <h2 className="font-display text-2xl lg:text-3xl font-bold mb-10">Kiến trúc xử lý</h2>

        <div className="flex flex-col lg:flex-row bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl overflow-hidden">
          {archSteps.map((step, i) => (
            <div
              key={step.num}
              className="flex-1 px-5 py-7 text-center relative border-b lg:border-b-0 lg:border-r border-[var(--border)] last:border-0"
            >
              <p className="font-mono text-[10px] text-[var(--text-3)] tracking-widest mb-2">{step.num}</p>
              <p className="font-display text-[13px] font-bold text-[var(--text-1)] mb-1">{step.name}</p>
              <p className="text-[11px] text-[var(--text-3)] leading-relaxed whitespace-pre-line">{step.detail}</p>
              {i < archSteps.length - 1 && (
                <div className="hidden lg:block absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 z-10
                  font-mono text-[var(--text-3)] text-[12px]">
                  →
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Team members */}
      <section className="max-w-6xl mx-auto px-6 py-16 border-t border-[var(--border)]">
        <div className="flex items-center gap-2 font-mono text-[11px] text-[var(--cyan)] tracking-[0.15em] uppercase mb-3">
          <span className="w-4 h-px bg-[var(--cyan)]" />
          Nhóm thực hiện
        </div>
        <h2 className="font-display text-2xl lg:text-3xl font-bold mb-10">Thành viên nhóm 8</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {members.map((m, i) => (
            <div
              key={m.mssv}
              className="relative bg-[var(--bg-surface)] border border-[var(--border)] rounded-2xl p-6
                hover:border-[var(--border-glow)] hover:bg-[var(--bg-card)] hover:-translate-y-0.5
                transition-all duration-200 overflow-hidden group"
            >
              <div
                className="absolute top-0 left-0 right-0 h-[2px] opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                style={{ background: 'linear-gradient(90deg, transparent, var(--cyan), transparent)' }}
              />

              {/* Avatar placeholder */}
              <div
                className="w-10 h-10 rounded-full flex items-center justify-center mb-4 font-display text-[15px] font-bold"
                style={{
                  background: `hsl(${(i * 67 + 190) % 360}, 60%, 12%)`,
                  border: `1px solid hsl(${(i * 67 + 190) % 360}, 60%, 30%)`,
                  color: `hsl(${(i * 67 + 190) % 360}, 80%, 65%)`,
                }}
              >
                {m.name.split(' ').pop()![0]}
              </div>

              <p className="font-display text-[14px] font-semibold text-[var(--text-1)] mb-1">{m.name}</p>
              <p className="font-mono text-[10px] text-[var(--cyan-dim)] tracking-widest">{m.mssv}</p>
            </div>
          ))}
        </div>
      </section>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={handleFileChange}
      />
      <Toast message={toast} onDone={() => setToast(null)} />
    </>
  )
}
