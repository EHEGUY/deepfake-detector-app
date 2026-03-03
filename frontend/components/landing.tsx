'use client'

import { ArrowRight } from 'lucide-react'

interface LandingProps {
  onLaunch: () => void
}

export default function Landing({ onLaunch }: LandingProps) {
  return (
    <div className="relative w-full min-h-screen bg-transparent overflow-hidden">

      {/* Main Content */}
      <main className="relative z-10 max-w-5xl mx-auto px-6 pt-32 pb-20 flex items-center min-h-screen">
        <div className="space-y-10">
          {/* Hero Content */}
          <div className="space-y-8">
            <h1 className="text-7xl lg:text-8xl font-black tracking-tighter leading-[0.95] text-white">
              SEE THROUGH<br />THE DECEPTION
            </h1>

            <p className="text-lg text-white/70 max-w-xl leading-relaxed font-light">
              Advanced neural network analysis detects AI-generated and manipulated images with forensic precision. Pixel-level artifact detection reveals what the human eye cannot see.
            </p>
          </div>

          {/* Dual Button Setup */}
          <div className="flex flex-col sm:flex-row gap-4 pt-4">
            <button
              onClick={onLaunch}
              className="group relative px-8 py-4 rounded-lg bg-[#0066cc] text-white font-bold text-base tracking-wide overflow-hidden transition-all duration-300 hover:bg-[#0052a3] hover:shadow-[0_0_50px_rgba(0,102,204,0.6)] hover:-translate-y-1"
            >
              <span className="relative z-10 flex items-center justify-center gap-3">
                GET STARTED
                <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
              </span>
            </button>
            
            <button
              className="px-8 py-4 rounded-lg border-2 border-white/30 text-white font-bold text-base tracking-wide transition-all duration-300 hover:border-white/60 hover:bg-white/5"
            >
              LEARN MORE
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}
