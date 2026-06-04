export interface PredictionResult {
  label: string        // "THẬT" | "GIẢ (AI)"
  label_en: string     // "Real" | "Fake"
  is_fake: boolean
  prob_fake: number    // 0–1
  prob_real: number    // 0–1
  confidence: number   // max(prob_real, prob_fake)
  logit: number
  inference_ms: number
  gradcam_b64: string | null
}

export interface HealthResponse {
  status: string
  checkpoint: string
  device: string
  torch: string
}

export type AppState = 'idle' | 'analyzing' | 'results' | 'error'
