import type { ButtonHTMLAttributes, ReactNode } from 'react'

type Variant = 'ghost' | 'outline' | 'primary'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  children: ReactNode
}

const base =
  'font-display text-[11px] font-semibold tracking-widest uppercase px-4 py-2 rounded-lg border transition-all duration-200 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed'

const variants: Record<Variant, string> = {
  ghost:
    'bg-transparent border-[var(--border)] text-[var(--text-2)] hover:border-[var(--cyan)] hover:text-[var(--cyan)] hover:bg-[rgba(0,200,255,0.06)]',
  outline:
    'bg-transparent border-[var(--cyan)] text-[var(--cyan)] hover:bg-[rgba(0,200,255,0.1)]',
  primary:
    'bg-[var(--cyan)] border-[var(--cyan)] text-[#05080f] hover:bg-[#33d6ff] hover:shadow-[0_0_20px_rgba(0,200,255,0.35)] active:scale-[0.98]',
}

export function Button({ variant = 'ghost', className = '', children, ...props }: Props) {
  return (
    <button className={`${base} ${variants[variant]} ${className}`} {...props}>
      {children}
    </button>
  )
}
