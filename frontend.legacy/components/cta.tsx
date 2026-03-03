export default function CTA({ onGetStarted }: { onGetStarted: () => void }) {
  return (
    <section className="py-24 px-6 bg-foreground text-background">
      <div className="max-w-4xl mx-auto text-center space-y-10">
        <div className="space-y-6">
          <h2 className="text-6xl lg:text-7xl font-black tracking-tight">
            Ready to Know the Truth?
          </h2>
          <p className="text-lg opacity-90 max-w-2xl mx-auto leading-relaxed">
            Start detecting deepfakes and synthetic media right now. No signup required.
          </p>
        </div>

        <button
          onClick={onGetStarted}
          className="px-8 py-3.5 bg-background text-foreground rounded-lg font-semibold hover:bg-opacity-90 transition-all active:scale-95 text-base inline-block"
        >
          Start Free Analysis
        </button>
      </div>
    </section>
  )
}
