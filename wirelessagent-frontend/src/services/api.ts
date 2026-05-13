import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'

const API_BASE_URL = 'http://localhost:8000'

export type StreamEvent =
  | { type: 'log'; log: { type: 'info' | 'success' | 'warning' | 'error'; message: string; time: string } }
  | { type: 'result'; result: any }
  | { type: 'progress'; processed: number; total: number; stage?: number; message?: string }
  | { type: 'complete'; total: number; success: number; failed: number }
  | { type: 'error'; message: string }

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 600000,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    this.client.interceptors.request.use(
      (config) => {
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    this.client.interceptors.response.use(
      (response) => {
        return response
      },
      (error) => {
        console.error('API Error:', error)
        return Promise.reject(error)
      }
    )
  }

  async uploadCSV(file: File, onProgress?: (progress: number) => void): Promise<any> {
    const formData = new FormData()
    formData.append('file', file)

    const config: AxiosRequestConfig = {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }

    if (onProgress) {
      config.onUploadProgress = (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      }
    }

    try {
      const response: AxiosResponse = await this.client.post('/upload-csv', formData, config)
      return response.data
    } catch (error) {
      console.error('CSV upload failed:', error)
      throw error
    }
  }

  async processCSV(file: File, useKnowledgeBase: boolean = false, onProgress?: (progress: number) => void): Promise<any> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('use_knowledge_base', useKnowledgeBase.toString())

    const config: AxiosRequestConfig = {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }

    if (onProgress) {
      config.onUploadProgress = (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      }
    }

    try {
      const response: AxiosResponse = await this.client.post('/process-csv', formData, config)
      return response.data
    } catch (error) {
      console.error('CSV processing failed:', error)
      throw error
    }
  }

  async processCSVStream(
    file: File,
    useKnowledgeBase: boolean = false,
    onEvent: (event: StreamEvent) => void
  ): Promise<void> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('use_knowledge_base', useKnowledgeBase.toString())

    const response = await fetch(`${API_BASE_URL}/process-csv-stream`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || `CSV streaming failed with status ${response.status}`)
    }

    if (!response.body) {
      throw new Error('当前浏览器不支持流式响应读取')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        onEvent(JSON.parse(trimmed) as StreamEvent)
      }
    }

    buffer += decoder.decode()
    const trimmed = buffer.trim()
    if (trimmed) {
      onEvent(JSON.parse(trimmed) as StreamEvent)
    }
  }

  async getResults(): Promise<any> {
    try {
      const response: AxiosResponse = await this.client.get('/results')
      return response.data
    } catch (error) {
      console.error('Failed to get results:', error)
      throw error
    }
  }

  async getHealth(): Promise<any> {
    try {
      const response: AxiosResponse = await this.client.get('/health', {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      return response.data
    } catch (error) {
      console.error('Health check failed:', error)
      throw error
    }
  }
}

export default new ApiService()
