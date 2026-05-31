import { create } from 'zustand'
import { predict } from '../api/predict'
import type { AppState, PredictionResult } from '../types/prediction'

interface AppStore {
  // state
  appState: AppState
  file: File | null
  previewUrl: string | null
  result: PredictionResult | null
  gradcamVisible: boolean
  errorMsg: string | null

  // actions
  setFile: (file: File) => void
  analyze: () => Promise<void>
  toggleGradcam: () => void
  reset: () => void
}

export const useAppStore = create<AppStore>((set, get) => ({
  appState: 'idle',
  file: null,
  previewUrl: null,
  result: null,
  gradcamVisible: false,
  errorMsg: null,

  setFile: (file) => {
    const prev = get().previewUrl
    if (prev) URL.revokeObjectURL(prev)
    set({
      file,
      previewUrl: URL.createObjectURL(file),
      appState: 'idle',
      result: null,
      gradcamVisible: false,
      errorMsg: null,
    })
  },

  analyze: async () => {
    const { file } = get()
    if (!file) return
    set({ appState: 'analyzing', result: null, errorMsg: null })
    try {
      const result = await predict(file, true)
      set({ appState: 'results', result })
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : 'Lỗi kết nối tới server.'
      set({ appState: 'error', errorMsg: msg })
    }
  },

  toggleGradcam: () => set((s) => ({ gradcamVisible: !s.gradcamVisible })),

  reset: () => {
    const prev = get().previewUrl
    if (prev) URL.revokeObjectURL(prev)
    set({
      appState: 'idle',
      file: null,
      previewUrl: null,
      result: null,
      gradcamVisible: false,
      errorMsg: null,
    })
  },
}))
