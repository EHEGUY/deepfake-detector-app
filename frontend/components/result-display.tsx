'use client'

import { useEffect, useState } from 'react'

interface ResultDisplayProps {
  prediction: {
    result: 'Fake' | 'Real'  //  CHANGED FROM prediction TO result
    confidence: number
  }
  imageUrl: string
  onReset: () => void
}

export default function ResultDisplay({ prediction, imageUrl, onReset }: ResultDisplayProps) {
  const [displayConfidence, setDisplayConfidence] = useState(0)

  useEffect(() => {
    // Animate the confidence bar
    const timer = setTimeout(() => {
      setDisplayConfidence(prediction.confidence)
    }, 300)
    return () => clearTimeout(timer)
  }, [prediction.confidence])

  //  UPDATED LOGIC TO USE .result
  const isFake = prediction.result === 'Fake' 
  
  const resultColor = isFake ? 'text-red-400' : 'text-green-400'
  const resultBg = isFake ? 'bg-red-500/10' : 'bg-green-500/10'
  const resultBorder = isFake ? 'border-red-500/30' : 'border-green-500/30'
  const badgeBg = isFake ? 'bg-red-500/20 text-red-300 border-red-500/30' : 'bg-green-500/20 text-green-300 border-green-500/30'
  const barColor = isFake ? 'bg-red-500' : 'bg-green-500'

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Image Display */}
      <div className="rounded-xl overflow-hidden border border-white/10 backdrop-blur shadow-2xl">
        <img src={imageUrl} alt="Analysis Target" className="w-full h-auto object-cover" />
      </div>

      {/* Result Card */}
      <div className={`border-2 ${resultBorder} ${resultBg} rounded-xl p-8 backdrop-blur-sm`}>
        <div className="space-y-6">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4 flex-1">
              <div className={`w-14 h-14 rounded-full flex items-center justify-center ${isFake ? 'bg-red-500/20' : 'bg-green-500/20'}`}>
                {isFake ? (
                  <svg className="w-7 h-7 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" />
                  </svg>
                ) : (
                  <svg className="w-7 h-7 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" />
                  </svg>
                )}
              </div>

              <div>
                <p className="text-xs text-white/50 uppercase tracking-widest font-semibold">Detection Result</p>
                <h2 className={`text-4xl md:text-5xl font-black ${resultColor}`}>
                  {prediction.result} {/*  UPDATED THIS LINE */}
                </h2>
              </div>
            </div>

            <div className={`px-3 py-1.5 rounded-full text-sm font-semibold border ${badgeBg}`}>
              {isFake ? 'Unverified' : 'Verified'}
            </div>
          </div>

          <div className="pt-4 border-t border-white/10 space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-xs text-white/50 uppercase tracking-widest font-semibold">Confidence Score</p>
              <p className={`text-3xl font-bold ${resultColor}`}>{displayConfidence.toFixed(1)}%</p>
            </div>

            <div className="relative h-3 rounded-full bg-white/10 overflow-hidden">
              <div
                className={`h-full ${barColor} transition-all duration-1000 ease-out`}
                style={{ width: `${displayConfidence}%` }}
              />
            </div>

            <p className="text-sm text-white/60 leading-relaxed pt-1">
              {isFake
                ? ' ALERT: Forensic analysis has identified synthetic artifacts consistent with AI generation.'
                : ' PROTECTED: Pixel consistency and noise distribution appear authentic.'}
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-3 justify-center">
        <button onClick={onReset} className="px-8 py-3 bg-white text-black rounded-lg font-semibold active:scale-95">
          Analyze Another Image
        </button>
      </div>
    </div>
  )
}