'use client'

import { useRef, useState } from 'react'

interface DragDropZoneProps {
  onFileSelect: (file: File) => void
}

export default function DragDropZone({ onFileSelect }: DragDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      onFileSelect(files[0])
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      onFileSelect(files[0])
    }
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className="cursor-pointer group"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInputChange}
        className="hidden"
      />

      <div
        onClick={() => fileInputRef.current?.click()}
        className={`relative p-12 md:p-16 rounded-2xl border-2 transition-all duration-300 backdrop-blur-sm ${
          isDragging
            ? 'border-white/60 bg-white/10 shadow-lg shadow-white/10'
            : 'border-white/20 bg-white/5 hover:border-white/40 hover:bg-white/8'
        }`}
      >
        <div className="flex flex-col items-center justify-center space-y-6">
          {/* Icon */}
          <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-300 ${
            isDragging
              ? 'bg-white/20 text-white'
              : 'bg-white/10 text-white/80 group-hover:bg-white/20 group-hover:text-white'
          }`}>
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>

          {/* Text Content */}
          <div className="text-center space-y-2">
            <h3 className="text-xl font-bold text-white">
              {isDragging ? 'Release to upload' : 'Drag & drop your image'}
            </h3>
            <p className="text-white/60 text-sm">or click to select a file</p>
            <p className="text-xs text-white/40 mt-3">
              JPG, PNG, GIF, WebP • Up to 50MB
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
