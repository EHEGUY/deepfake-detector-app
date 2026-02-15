'use client'

export default function LoadingState() {
  return (
    <div className="space-y-8 animate-fade-in">
      {/* Image Placeholder */}
      <div className="rounded-xl border border-white/10 bg-white/5 backdrop-blur h-96 flex items-center justify-center overflow-hidden">
        <div className="text-center space-y-4">
          <div className="w-12 h-12 rounded-full border-2 border-t-white border-l-white border-r-white/20 border-b-white/20 mx-auto animate-spin" />
          <p className="text-white/60 text-sm">Processing image...</p>
        </div>
      </div>

      {/* Result Card Skeleton */}
      <div className="bg-white/5 backdrop-blur border border-white/10 rounded-xl p-8 md:p-10">
        <div className="space-y-6">
          {/* Header Skeleton */}
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-full bg-white/10 animate-pulse" />
              <div className="space-y-2">
                <div className="h-3 w-20 bg-white/10 rounded animate-pulse" />
                <div className="h-6 w-32 bg-white/10 rounded animate-pulse" />
              </div>
            </div>
            <div className="h-8 w-20 bg-white/10 rounded-full animate-pulse" />
          </div>

          {/* Confidence Section Skeleton */}
          <div className="pt-4 border-t border-white/10 space-y-3">
            <div className="flex items-center justify-between">
              <div className="h-3 w-24 bg-white/10 rounded animate-pulse" />
              <div className="h-6 w-16 bg-white/10 rounded animate-pulse" />
            </div>
            <div className="h-3 w-full bg-white/10 rounded-full animate-pulse" />
            <div className="h-4 w-48 bg-white/10 rounded animate-pulse" />
          </div>
        </div>
      </div>

      {/* Button Skeleton */}
      <div className="flex gap-3 justify-center">
        <div className="px-8 py-3 w-48 bg-white/10 rounded-lg animate-pulse" />
        <div className="px-8 py-3 w-40 border border-white/10 rounded-lg bg-white/5 animate-pulse" />
      </div>

      {/* Loading Text */}
      <div className="text-center pt-2">
        <p className="text-white/60 text-sm">Analyzing with advanced AI...</p>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fadeIn 0.6s ease-out;
        }
      `}</style>
    </div>
  )
}
