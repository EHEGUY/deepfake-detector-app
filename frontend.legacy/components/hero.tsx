export default function Hero({ onGetStarted }: { onGetStarted: () => void }) {
  return (
    <section className="min-h-screen pt-32 pb-20 px-6 flex items-center justify-center bg-background">
      <div className="max-w-4xl mx-auto text-center space-y-10">
        <div className="space-y-6">
          <p className="text-sm font-semibold tracking-widest text-muted-foreground uppercase">
            Advanced Detection Technology
          </p>
          <h1 className="text-7xl lg:text-8xl font-black tracking-tight leading-tight text-foreground">
            See Through the Deception
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Détecteur uses cutting-edge AI to identify deepfakes and synthetic media with unprecedented accuracy. Protect yourself from digital manipulation.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
          <button
            onClick={onGetStarted}
            className="px-8 py-3.5 bg-foreground text-background rounded-lg font-semibold hover:bg-opacity-90 transition-all active:scale-95 text-base"
          >
            Try Now
          </button>
          <button
            onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
            className="px-8 py-3.5 bg-transparent text-foreground border-2 border-foreground rounded-lg font-semibold hover:bg-foreground hover:text-background transition-all active:scale-95 text-base"
          >
            Learn More
          </button>
        </div>
      </div>
    </section>
  )
}
