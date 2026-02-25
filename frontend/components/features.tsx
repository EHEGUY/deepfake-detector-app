export default function Features() {
  const features = [
    {
      title: 'Lightning Fast',
      description: 'Get results in seconds. Real-time analysis powered by advanced neural networks.'
    },
    {
      title: 'Highly Accurate',
      description: 'Trained on thousands of images for precision detection.'
    },
    {
      title: 'Privacy First',
      description: 'Your images are never stored or shared. All analysis happens locally on your device.'
    },
    {
      title: 'Detailed Insights',
      description: 'Understand exactly why an image is classified as real or synthetic with confidence scores.'
    }
  ]

  return (
    <section id="features" className="py-24 px-6 bg-secondary">
      <div className="max-w-6xl mx-auto space-y-20">
        <div className="text-center space-y-4">
          <h2 className="text-6xl lg:text-7xl font-black tracking-tight">
            Key Features
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Built with precision and designed for clarity.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, i) => (
            <div key={i} className="bg-background p-8 rounded-xl border border-border hover:border-foreground hover:shadow-lg transition-all duration-300">
              <h3 className="text-2xl font-bold mb-3 text-foreground">{feature.title}</h3>
              <p className="text-base text-muted-foreground leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
