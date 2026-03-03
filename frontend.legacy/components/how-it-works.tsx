export default function HowItWorks() {
  const steps = [
    {
      number: '1',
      title: 'Upload Image',
      description: 'Choose any image from your device or paste a URL to analyze.'
    },
    {
      number: '2',
      title: 'AI Analysis',
      description: 'Our neural network processes the image with advanced deepfake detection algorithms.'
    },
    {
      number: '3',
      title: 'Instant Results',
      description: 'Receive a detailed verdict with confidence score and analysis breakdown.'
    }
  ]

  return (
    <section className="py-24 px-6 bg-background">
      <div className="max-w-6xl mx-auto space-y-20">
        <div className="text-center space-y-4">
          <h2 className="text-6xl lg:text-7xl font-black tracking-tight">
            How It Works
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Simple, intuitive, and powerful. Three steps to truth.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step, i) => (
            <div key={i} className="relative">
              {i < steps.length - 1 && (
                <div className="hidden md:block absolute top-6 left-[60%] w-1/3 h-px bg-border"></div>
              )}
              <div className="space-y-5">
                <div className="w-14 h-14 rounded-full bg-foreground text-background flex items-center justify-center font-black text-lg">
                  {step.number}
                </div>
                <div>
                  <h3 className="text-xl font-bold mb-2 text-foreground">{step.title}</h3>
                  <p className="text-base text-muted-foreground leading-relaxed">{step.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
