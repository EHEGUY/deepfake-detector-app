'use client'

import { useState } from 'react'
import { Background3D } from '@/components/3d-background'
import { Section1, Section2, Section3, Section4, SectionsWrapper } from '@/components/landing-sections'
import DeepfakeDetector from '@/components/deepfake-detector'

export default function Home() {
  const [showDetector, setShowDetector] = useState(false)

  if (showDetector) {
    return (
      <main className="min-h-screen bg-background">
        <DeepfakeDetector onBack={() => setShowDetector(false)} />
      </main>
    )
  }

  return (
    <main className="app bg-black text-white overflow-x-hidden min-h-screen">
      <Background3D />

      <style>{`
        @import url("https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap");
        
        .app {
          font-family: "Inter", sans-serif;
          background-color: #0a0a0a;
          color: #f5f5f5;
          overflow-x: hidden;
          line-height: 1.5;
        }

        .content {
          position: relative;
          z-index: 2;
        }

        .section {
          height: 100vh;
          width: 100%;
          display: flex;
          align-items: center;
          position: relative;
        }

        .section-inner {
          width: 100%;
          padding: 0 5%;
          max-width: 1600px;
          margin: 0 auto;
        }

        .title {
          font-size: clamp(3rem, 12vw, 10rem);
          line-height: 1.1;
          margin-bottom: 1.5rem;
          font-weight: 900;
          opacity: 0;
          transform: translateY(50px);
          text-transform: uppercase;
          color: #f5f5f5;
          letter-spacing: -0.02em;
        }

        .description {
          font-size: 1.25rem;
          max-width: 600px;
          margin-bottom: 2rem;
          opacity: 0;
          transform: translateY(30px);
          color: rgba(245, 245, 245, 0.7);
          font-weight: 300;
        }

        @media (max-width: 768px) {
          .title {
            font-size: clamp(2.5rem, 10vw, 6rem);
          }
          .description {
            font-size: 1.1rem;
          }
        }
      `}</style>

      <SectionsWrapper>
        <div className="content">
          <Section1 onGetStarted={() => setShowDetector(true)} />
          <Section2 />
          <Section3 />
          <Section4 onGetStarted={() => setShowDetector(true)} />
        </div>
      </SectionsWrapper>

      <footer className="fixed bottom-4 left-4 text-sm text-white/50 z-50">
        <p>© 2024 Détecteur</p>
      </footer>
    </main>
  )
}
