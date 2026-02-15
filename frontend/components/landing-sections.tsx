'use client'

import { useScrollAnimations } from '@/hooks/use-scroll-animations'

function SectionsWrapper({ children }: { children: React.ReactNode }) {
  useScrollAnimations()
  return <>{children}</>
}

export function Section1({ onGetStarted }: { onGetStarted: () => void }) {
  return (
    <section className="section h-screen w-full flex items-center relative">
      <div className="section-inner w-full px-[5%] max-w-6xl mx-auto pointer-events-auto relative z-10">
        <h1 className="title text-8xl font-black leading-tight mb-8 opacity-0 translate-y-[50px] uppercase tracking-tight">
          See Through the Deception
        </h1>
        <p className="description text-2xl max-w-2xl mb-12 opacity-0 translate-y-[30px] text-white/70 font-light">
          Détecteur uses cutting-edge AI to identify deepfakes and synthetic media with unprecedented accuracy.
        </p>
        <button
          onClick={onGetStarted}
          className="px-8 py-4 bg-white text-black rounded-lg font-semibold hover:bg-opacity-90 transition-all active:scale-95 relative z-20 pointer-events-auto"
        >
          Try Now
        </button>
      </div>
    </section>
  )
}

export function Section2() {
  return (
    <section className="section h-screen w-full flex items-center relative">
      <div className="section-inner w-full px-[5%] max-w-6xl mx-auto pointer-events-auto relative z-10">
        <h1 className="title text-8xl font-black leading-tight mb-8 opacity-0 translate-y-[50px] uppercase tracking-tight">
          Key Features
        </h1>
        <p className="description text-2xl max-w-2xl mb-12 opacity-0 translate-y-[30px] text-white/70 font-light">
          Advanced detection technology built for precision and clarity.
        </p>

        <div className="grid md:grid-cols-2 gap-8 max-w-3xl">
          {[
            { title: 'Lightning Fast', desc: 'Real-time analysis in seconds' },
            { title: 'Highly Accurate', desc: 'Trained on millions of images' },
            { title: 'Privacy First', desc: 'Local processing, no storage' },
            { title: 'Detailed Insights', desc: 'Confidence scores and analysis' }
          ].map((feature, i) => (
            <div
              key={i}
              className="bg-white/5 backdrop-blur-sm p-6 rounded-xl border border-white/10 hover:border-red-500/30 transition-colors"
            >
              <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
              <p className="text-white/60 text-sm">{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export function Section3() {
  return (
    <section className="section h-screen w-full flex items-center relative">
      <div className="section-inner w-full px-[5%] max-w-6xl mx-auto pointer-events-auto relative z-10">
        <h1 className="title text-8xl font-black leading-tight mb-8 opacity-0 translate-y-[50px] uppercase tracking-tight">
          How It Works
        </h1>

        <div className="grid md:grid-cols-3 gap-8 max-w-4xl">
          {[
            { num: '01', title: 'Upload', desc: 'Drag and drop your image' },
            { num: '02', title: 'Analyze', desc: 'AI processes in real-time' },
            { num: '03', title: 'Results', desc: 'Get detailed confidence score' }
          ].map((step, i) => (
            <div key={i} className="relative">
              <div className="text-6xl font-black text-white/20 mb-4">{step.num}</div>
              <h3 className="text-2xl font-bold mb-3">{step.title}</h3>
              <p className="text-white/70">{step.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export function Section4({ onGetStarted }: { onGetStarted: () => void }) {
  return (
    <section className="section h-screen w-full flex items-center relative">
      <div className="section-inner w-full px-[5%] max-w-6xl mx-auto pointer-events-auto relative z-10 text-center">
        <h1 className="title text-8xl font-black leading-tight mb-8 opacity-0 translate-y-[50px] uppercase tracking-tight">
          Ready to Know the Truth?
        </h1>
        <p className="description text-2xl max-w-2xl mx-auto mb-12 opacity-0 translate-y-[30px] text-white/70 font-light">
          Start detecting deepfakes and synthetic media right now. No signup required.
        </p>
        <button
          onClick={onGetStarted}
          className="px-8 py-4 bg-white text-black rounded-lg font-semibold hover:bg-opacity-90 transition-all active:scale-95 relative z-20 pointer-events-auto"
        >
          Start Free Analysis
        </button>
      </div>
    </section>
  )
}

export { SectionsWrapper }
