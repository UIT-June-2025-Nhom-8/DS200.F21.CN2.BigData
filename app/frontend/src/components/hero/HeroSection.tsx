import { ScanAnimation } from './ScanAnimation'

const metrics = [
  { val: '97.33%', label: 'Accuracy' },
  { val: '97.45%', label: 'F1 Score' },
  { val: '99.70%', label: 'AUC-ROC' },
]

export function HeroSection() {
  return (
    <section className="max-w-6xl mx-auto px-6 pt-16 pb-12 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
      {/* Left */}
      <div className="animate-fade-in-up">
        <div className="flex items-center gap-2 font-mono text-[11px] text-[var(--cyan)] tracking-[0.15em] uppercase mb-4">
          <span className="inline-block w-6 h-px bg-[var(--cyan)]" />
          DS200.F21.CN2 · UIT-VNU HCM · 2025
        </div>

        <h1 className="font-display text-4xl lg:text-5xl font-bold leading-[1.15] tracking-tight mb-5">
          Phát hiện<br />
          <em className="not-italic text-[var(--cyan)]">khuôn mặt AI</em><br />
          với độ chính xác cao
        </h1>

        <p className="text-[var(--text-2)] text-[15px] leading-relaxed mb-8">
          Mô hình phân loại nhị phân dựa trên EfficientNet-B0 được huấn luyện
          trên hơn <strong className="text-[var(--text-1)]">280,000 ảnh</strong> từ 4 bộ
          dữ liệu thực tế. Phân biệt khuôn mặt thật và khuôn mặt do AI tạo ra
          chỉ trong vài giây.
        </p>

        <div className="grid grid-cols-3 gap-3">
          {metrics.map(({ val, label }) => (
            <div
              key={label}
              className="bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-4 text-center
                hover:border-[var(--border-glow)] transition-colors duration-200"
            >
              <span className="font-display text-xl font-bold text-[var(--cyan)] block">{val}</span>
              <span className="font-mono text-[9px] text-[var(--text-3)] tracking-widest uppercase mt-1 block">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Right */}
      <div className="hidden lg:flex items-center justify-center animate-fade-in">
        <ScanAnimation />
      </div>
    </section>
  )
}
