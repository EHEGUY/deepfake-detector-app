export default function Header({ onGetStarted }: { onGetStarted: () => void }) {
  return (
    <header className="fixed top-0 w-full bg-background/95 backdrop-blur-md z-50 border-b border-border">
      <nav className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">
        <div className="text-2xl font-black tracking-tight text-foreground">Détecteur</div>
        <button
          onClick={onGetStarted}
          className="px-6 py-2 bg-foreground text-background rounded-lg font-semibold text-sm hover:bg-opacity-90 transition-all active:scale-95"
        >
          Get Started
        </button>
      </nav>
    </header>
  )
}
