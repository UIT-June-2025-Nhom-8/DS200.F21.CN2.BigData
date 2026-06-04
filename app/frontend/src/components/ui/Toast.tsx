import { useEffect, useState } from 'react'

interface ToastProps {
  message: string | null
  onDone: () => void
}

export function Toast({ message, onDone }: ToastProps) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (!message) return
    setVisible(true)
    let inner: ReturnType<typeof setTimeout>
    const outer = setTimeout(() => {
      setVisible(false)
      inner = setTimeout(onDone, 300)
    }, 3000)
    return () => {
      clearTimeout(outer)
      clearTimeout(inner)
    }
  }, [message, onDone])

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 font-mono text-[12px] text-[var(--cyan)]
        bg-[var(--bg-card)] border border-[var(--border-glow)] rounded-lg px-5 py-3
        shadow-[0_8px_32px_rgba(0,0,0,0.5)]
        transition-all duration-300
        ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8 pointer-events-none'}`}
    >
      {message}
    </div>
  )
}
