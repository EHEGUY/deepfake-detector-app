interface HeaderProps {
  activeView: 'landing' | 'detector' | 'about'
  onViewChange: (view: 'landing' | 'detector' | 'about') => void
}

export default function Header({
  activeView,
  onViewChange,
}: HeaderProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-black/10 border-b border-white/10">
      <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
        <button
          onClick={() => onViewChange('landing')}
          className="text-sm font-black tracking-wider text-white hover:text-white/80 transition-colors"
        >
          DÉTECTEUR
        </button>
        
        <nav className="flex items-center gap-2">
          <button
            onClick={() => onViewChange('detector')}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-300 ${
              activeView === 'detector'
                ? 'bg-[#0066cc] text-white'
                : 'text-white hover:bg-white/10'
            }`}
          >
            Detector
          </button>
          <button
            onClick={() => onViewChange('about')}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-300 ${
              activeView === 'about'
                ? 'bg-[#0066cc] text-white'
                : 'text-white hover:bg-white/10'
            }`}
          >
            About
          </button>
        </nav>
      </div>
    </header>
  )
}
