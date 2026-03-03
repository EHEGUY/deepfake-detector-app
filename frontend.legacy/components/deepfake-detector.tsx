'use client'

import { useState, useRef } from 'react'
import { Background3D } from './3d-background'
import DragDropZone from './drag-drop-zone'
import ResultDisplay from './result-display'
import LoadingState from './loading-state'

interface PredictionResult {
  prediction: 'Fake' | 'Real'
  confidence: number
}

export default function DeepfakeDetector({ onBack }: { onBack: () => void }) {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file')
      return
    }

    try {
      setError(null)
      setLoading(true)

      const reader = new FileReader()
      reader.onloadend = () => {
        setUploadedImage(reader.result as string)
      }
      reader.readAsDataURL(file)

      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Prediction request failed')
      }

      const result: PredictionResult = await response.json()
      setPrediction(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during prediction')
      setPrediction(null)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setUploadedImage(null)
    setPrediction(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-black text-white relative overflow-hidden">
      <Background3D />
      
      {/* Header */}
      <header className="sticky top-0 bg-black/50 backdrop-blur-md z-50 border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-black tracking-tight">Détecteur</h1>
          <button
            onClick={onBack}
            className="text-white/60 hover:text-white transition-colors text-sm font-semibold"
          >
            ← Back
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center p-4 md:p-8 relative z-10">
        <div className="w-full max-w-2xl">
          {!uploadedImage ? (
            <div className="space-y-8 animate-fade-in">
              <div className="text-center space-y-3 mb-8">
                <h2 className="text-4xl md:text-5xl font-black tracking-tight text-white">Analyze Your Image</h2>
                <p className="text-white/60">Upload an image to detect if it's real or synthetic</p>
              </div>
              <DragDropZone onFileSelect={handleFileUpload} />
            </div>
          ) : (
            <div className="space-y-8 animate-fade-in">
              {loading && <LoadingState />}
              {!loading && prediction && (
                <ResultDisplay prediction={prediction} imageUrl={uploadedImage} onReset={handleReset} />
              )}
              {error && (
                <div className="bg-red-500/10 backdrop-blur border border-red-500/20 rounded-xl p-4 text-red-400 text-center">
                  {error}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fadeIn 0.6s ease-out;
        }
      `}</style>
    </div>
  )
}
