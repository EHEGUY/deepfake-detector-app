export default function Footer() {
  return (
    <footer className="bg-secondary border-t border-border py-16 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="space-y-10">
          <div className="text-center space-y-3">
            <h3 className="font-black text-3xl text-foreground">Détecteur</h3>
            <p className="text-muted-foreground text-base">Advanced deepfake detection technology.</p>
          </div>

          <div className="text-center">
            <h4 className="font-bold text-sm uppercase tracking-wide text-muted-foreground mb-3">Contact</h4>
            <p>
              <a href="mailto:" className="hover:text-foreground transition-colors font-semibold text-foreground">your-email@example.com</a>
            </p>
          </div>
        </div>

        <div className="border-t border-border mt-10 pt-8 text-center text-muted-foreground text-sm">
          <p>© 2024 Détecteur. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}
