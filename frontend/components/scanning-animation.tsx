'use client'

import { useEffect, useState } from 'react'

export default function ScanningAnimation() {
  // Animated dots for the "Scanning..." text
  const [dots, setDots] = useState('')

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.')
    }, 400)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="absolute inset-0 z-20 pointer-events-none rounded-xl overflow-hidden">
      <style>{`
        @keyframes sleek-scan {
          0% { top: -10%; opacity: 0; }
          15% { opacity: 1; }
          85% { opacity: 1; }
          100% { top: 110%; opacity: 0; }
        }
        .animate-sleek-sweep {
          /* cubic-bezier makes it speed up in the middle and slow down at the edges */
          animation: sleek-scan 2.5s cubic-bezier(0.65, 0, 0.35, 1) infinite;
        }
      `}</style>

      {/* Very subtle dark tint to contrast the laser */}
      <div className="absolute inset-0 bg-black/10" />

      {/* Minimalist 1px Corner Reticles */}
      <div className="absolute top-6 left-6 w-8 h-8 border-t border-l border-white/40" />
      <div className="absolute top-6 right-6 w-8 h-8 border-t border-r border-white/40" />
      <div className="absolute bottom-6 left-6 w-8 h-8 border-b border-l border-white/40" />
      <div className="absolute bottom-6 right-6 w-8 h-8 border-b border-r border-white/40" />

      {/* The Sleek Laser */}
      <div className="absolute left-0 right-0 animate-sleek-sweep flex flex-col items-center">
        {/* Soft, faint trail */}
        <div className="w-full h-16 bg-gradient-to-b from-transparent to-[#0066cc]/10" />
        {/* Razor-thin laser core */}
        <div className="w-full h-[1px] bg-[#0066cc] shadow-[0_0_12px_1px_rgba(0,102,204,0.6)]" />
      </div>

      {/* Strictly "Scanning" Text in a glassmorphic pill */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="bg-black/40 backdrop-blur-md px-6 py-2 rounded-full border border-white/10 shadow-2xl">
          <span className="text-xs uppercase tracking-[0.3em] text-white font-medium w-[100px] inline-block text-left pl-2">
            Scanning{dots}
          </span>
        </div>
      </div>
    </div>
  )
}