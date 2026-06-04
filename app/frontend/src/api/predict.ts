import axios from 'axios'
import type { HealthResponse, PredictionResult } from '../types/prediction'

const api = axios.create({ baseURL: '/api' })

export async function predict(
  file: File,
  includeGradcam = true,
): Promise<PredictionResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<PredictionResult>(
    `/predict?gradcam=${includeGradcam}`,
    form,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  )
  return data
}

export async function healthCheck(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health')
  return data
}
