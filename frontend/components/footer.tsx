import { Github } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="fixed bottom-0 left-0 right-0 backdrop-blur-xl bg-black/20 border-t border-white/10 transition-colors duration-200">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <p className="text-sm text-white/70 font-medium">
          Project by SIDDT
        </p>
        <a
          href="https://github.com/EHEGUY"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-white/70 hover:text-[#0066cc] transition-colors duration-200 font-medium"
        >
          <Github size={18} />
          <span>EHEGUY</span>
        </a>
      </div>
    </footer>
  )
}
