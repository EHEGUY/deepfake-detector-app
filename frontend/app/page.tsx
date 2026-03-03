'use client'

import { useState } from 'react'
import Header from '@/components/header'
import Landing from '@/components/landing'
import Detector from '@/components/detector'
import About from '@/components/about'
import StarfieldBackground from '@/components/starfield-background'
import Footer from '@/components/footer'

export default function Home() {
  const [activeView, setActiveView] = useState<'landing' | 'detector' | 'about'>('landing')

  return (
    <div className="relative min-h-screen bg-transparent">
      <StarfieldBackground />
      <div className="relative z-10">
        {activeView !== 'landing' && (
          <Header 
            activeView={activeView} 
            onViewChange={setActiveView}
          />
        )}
        <main className={activeView === 'landing' ? '' : 'pt-24 pb-24'}>
          {activeView === 'landing' && (
            <Landing onLaunch={() => setActiveView('detector')} />
          )}
          {activeView === 'detector' && (
            <Detector />
          )}
          {activeView === 'about' && (
            <About />
          )}
        </main>
        {activeView !== 'landing' && <Footer />}
      </div>
    </div>
  )
}
