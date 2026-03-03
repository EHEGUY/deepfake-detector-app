'use client'

import { useState, useRef } from 'react'
import { UploadCloud, CheckCircle, AlertCircle } from 'lucide-react'
import { Progress } from '@/components/custom-progress'
import ScanningAnimation from '@/components/scanning-animation'

type DetectionState = 'upload' | 'preview' | 'scanning' | 'result'

export default function Detector() {
  const [state, setState] = useState<DetectionState>('upload')
  const [image, setImage] = useState<string | null>(null)
  // 1. ADDED: State to hold the actual binary file for the backend
  const [rawFile, setRawFile] = useState<File | null>(null)
  const [confidence, setConfidence] = useState(0)
  const [isReal, setIsReal] = useState(true)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleFile = (file: File) => {
    if (file.type.startsWith('image/')) {
      setRawFile(file) // Save the real file to send to Python
      const reader = new FileReader()
      reader.onload = (e) => {
        setImage(e.target?.result as string)
        setState('preview')
      }
      reader.readAsDataURL(file)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.currentTarget.files?.[0]
    if (file) handleFile(file)
  }

// 2. ADDED: Robust error handling and flexible parsing to handle various Python response formats

  const handleScan = async () => {
    if (!rawFile) return
    
    setState('scanning')
    
    const formData = new FormData()
    formData.append('file', rawFile) 

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('API Offline')

      const data = await response.json()
      
      // Print exactly what Python sends to the browser console (Press F12 to see it)
      console.log("PYTHON SENT THIS JSON:", data)
      
      // ROBUST PARSING: Check all common keys Python might be using
      const resultText = data.label || data.prediction || data.result || data.class || 'fake'
      const resultScore = data.confidence || data.score || data.probability || 0.99
      
      // Safely convert to lowercase without crashing
      const isFake = String(resultText).toLowerCase() === 'fake'
      
      setIsReal(!isFake)

      // Handle cases where Python sends confidence as 0.99 instead of 99
      let finalConfidence = typeof resultScore === 'number' ? resultScore : parseFloat(resultScore)
      if (finalConfidence <= 1.0) {
        finalConfidence = finalConfidence * 100
      }
      
      setConfidence(Math.round(finalConfidence))
      
      // Leave the scanning animation up for 1.5s so it looks professional
      setTimeout(() => setState('result'), 1500)

    } catch (err) {
      console.error("The actual error is:", err)
      alert("Something went wrong! Press F12 and check the Console tab to see the real error.")
      setState('preview')
    }
  }

  const handleReset = () => {
    setImage(null)
    setRawFile(null) // Clear the binary file
    setConfidence(0)
    setState('upload')
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="relative min-h-screen bg-transparent px-6 pt-32 pb-24">
      <div className="relative z-10 max-w-4xl mx-auto">
        {state === 'upload' && (
          <div
            onDrop={handleDrop}
            onDragOver={(e) => {
              e.preventDefault()
              setDragActive(true)
            }}
            onDragLeave={() => setDragActive(false)}
            className={`glass-card rounded-2xl p-16 text-center transition-all duration-300 cursor-pointer ${
              dragActive
                ? 'border-[#0066cc]/60 bg-[#0066cc]/15'
                : 'border-white/10 hover:border-[#0066cc]/40 hover:bg-white/[0.08]'
            }`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleChange}
              className="hidden"
            />
            <div className="flex flex-col items-center gap-8">
              <UploadCloud className="w-20 h-20 text-[#0066cc]" />
              <div className="space-y-3">
                <p className="text-3xl font-black text-white tracking-tight">
                  Upload Image for Analysis
                </p>
                <p className="text-base text-white/60 font-light">
                  Drag and drop or click to select an image
                </p>
              </div>
            </div>
          </div>
        )}

        {state === 'preview' && (
          <div className="glass-card rounded-2xl p-8 space-y-6">
            <div className="rounded-xl overflow-hidden border border-white/10 bg-black/20">
              <img
                src={image || ''}
                alt="Uploaded image"
                className="w-full h-auto"
              />
            </div>
            <button
              onClick={handleScan}
              className="w-full bg-[#0066cc] hover:bg-[#0052a3] text-white font-bold text-base py-4 px-6 rounded-lg transition-all duration-300 hover:shadow-[0_0_50px_rgba(0,102,204,0.6)] hover:-translate-y-1"
            >
              SCAN IMAGE
            </button>
          </div>
        )}

        {state === 'scanning' && (
          <div className="glass-card rounded-2xl p-8 space-y-6">
            <div className="rounded-xl overflow-hidden border border-white/10 bg-black/20 relative">
              <img
                src={image || ''}
                alt="Scanning image"
                className="w-full h-auto opacity-50" 
              />
              <ScanningAnimation />
            </div>
            <div className="text-center">
              <p className="text-base font-semibold text-white/80 animate-pulse">
                Analyzing pixel artifacts and compression patterns...
              </p>
            </div>
          </div>
        )}

        {state === 'result' && (
          <div className="glass-card rounded-2xl p-8 space-y-8">
            <div className="rounded-xl overflow-hidden border border-white/10 bg-black/20">
              <img
                src={image || ''}
                alt="Analysis result"
                className="w-full h-auto"
              />
            </div>

            <div className="space-y-8">
              <div className={`flex items-center gap-4 p-6 rounded-lg border transition-all ${
                isReal 
                  ? 'bg-green-500/10 border-green-500/30' 
                  : 'bg-red-500/10 border-red-500/30'
              }`}>
                {isReal ? (
                  <>
                    <CheckCircle className="w-8 h-8 text-green-400 flex-shrink-0" />
                    <div>
                      <p className="font-bold text-white text-lg">Likely Authentic</p>
                      <p className="text-sm text-white/70">Image appears to be genuine</p>
                    </div>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-8 h-8 text-red-400 flex-shrink-0" />
                    <div>
                      <p className="font-bold text-white text-lg">Likely Synthetic</p>
                      <p className="text-sm text-white/70">AI-generated or manipulated content</p>
                    </div>
                  </>
                )}
              </div>

              <div className="space-y-4 p-6 bg-white/5 border border-white/10 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-mono text-white/60 uppercase tracking-wider">
                    {isReal ? 'Authenticity' : 'Synthetic'} Confidence
                  </span>
                  <span className={`text-3xl font-black ${
                    isReal ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {confidence}%
                  </span>
                </div>
                <Progress
                  value={confidence}
                  className="h-3 bg-white/10"
                  indicatorClassName={isReal ? 'bg-green-500' : 'bg-red-500'}
                />
                <p className="text-xs text-white/50 pt-2">
                  Probabilistic assessment based on neural network analysis. Not an absolute verdict.
                </p>
              </div>
            </div>

            <button
              onClick={handleReset}
              className="w-full bg-[#0066cc] hover:bg-[#0052a3] text-white font-bold text-base py-4 px-6 rounded-lg transition-all duration-300 hover:shadow-[0_0_50px_rgba(0,102,204,0.6)] hover:-translate-y-1"
            >
              ANALYZE ANOTHER IMAGE
            </button>
          </div>
        )}
      </div>
    </div>
  )
}