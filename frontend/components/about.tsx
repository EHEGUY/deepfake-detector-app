export default function About() {
  return (
    <div className="relative min-h-screen bg-transparent px-6 py-12">
      <div className="relative z-10 max-w-3xl mx-auto">
        <div className="glass-card rounded-2xl p-8 md:p-12">
          <article className="space-y-8">
            <div>
              <h1 className="text-5xl md:text-6xl font-black text-white mb-6 leading-tight">
                About the Detector
              </h1>
              <p className="text-lg text-white/70 leading-relaxed">
                This tool is designed to assist in identifying AI-generated or manipulated images. It utilizes a fine-tuned ResNet18 neural network to analyze pixel-level inconsistencies and compression artifacts that are often invisible to the human eye. Built for transparency in digital forensics, this engine provides a probabilistic assessment based on its training data, not an absolute verdict.
              </p>
            </div>

            <div className="space-y-4 pt-8 border-t border-white/10">
              <h2 className="text-2xl font-bold text-white">
                Technical Approach
              </h2>
              <p className="text-white/70 leading-relaxed">
                The detection engine employs advanced neural network analysis to identify subtle patterns characteristic of synthetic media. By examining compression artifacts, frequency domain anomalies, and pixel-level inconsistencies, the model learns to distinguish between authentic and AI-generated content with high precision.
              </p>
            </div>

            <div className="space-y-4 pt-8 border-t border-white/10">
              <h2 className="text-2xl font-bold text-white">
                Limitations
              </h2>
              <ul className="space-y-3 text-white/70">
                <li className="flex gap-3">
                  <span className="text-[#0066cc] font-bold flex-shrink-0">•</span>
                  <span>Detection accuracy depends on image quality and compression level</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-[#0066cc] font-bold flex-shrink-0">•</span>
                  <span>Very high-quality synthetic images may evade detection</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-[#0066cc] font-bold flex-shrink-0">•</span>
                  <span>Results should be considered one component of a broader verification process</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-[#0066cc] font-bold flex-shrink-0">•</span>
                  <span>Regular model updates are necessary as AI generation techniques evolve</span>
                </li>
              </ul>
            </div>

            <div className="space-y-4 pt-8 border-t border-white/10">
              <h2 className="text-2xl font-bold text-white">
                Responsible Use
              </h2>
              <p className="text-white/70 leading-relaxed">
                This tool is intended for educational purposes and to assist in media verification workflows. Users should not rely solely on automated detection for critical decisions. Always employ multiple verification methods and consult with domain experts when handling sensitive content.
              </p>
            </div>
          </article>
        </div>
      </div>
    </div>
  )
}
