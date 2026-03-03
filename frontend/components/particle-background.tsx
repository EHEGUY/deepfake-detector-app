'use client'

import { useEffect, useRef } from 'react'

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  brightness: number
}

export default function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mouseRef = useRef({ x: 0, y: 0 })
  const particlesRef = useRef<Particle[]>([])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const setCanvasSize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    setCanvasSize()

    // Initialize particles
    const initParticles = () => {
      particlesRef.current = []
      const particleCount = 350
      for (let i = 0; i < particleCount; i++) {
        particlesRef.current.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          radius: Math.random() * 1.5 + 0.5,
          brightness: Math.random() * 0.5 + 0.5,
        })
      }
    }
    initParticles()

    // Mouse tracking
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }

    window.addEventListener('mousemove', handleMouseMove)

    // Animation loop
    const animate = () => {
      const isDark = document.documentElement.classList.contains('dark')
      
      // Clear canvas with background color
      ctx.fillStyle = isDark ? '#1d1d1f' : '#FFFFFF'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      const particles = particlesRef.current
      const mouse = mouseRef.current

      particles.forEach((particle) => {
        // Calculate distance to mouse
        const dx = mouse.x - particle.x
        const dy = mouse.y - particle.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        // Gravity pull towards mouse
        const maxDistance = 300
        if (distance < maxDistance) {
          const force = (1 - distance / maxDistance) * 0.3
          const angle = Math.atan2(dy, dx)
          particle.vx += Math.cos(angle) * force
          particle.vy += Math.sin(angle) * force
        }

        // Damping
        particle.vx *= 0.96
        particle.vy *= 0.96

        // Soft drift when far from mouse
        if (distance > maxDistance) {
          particle.vx += (Math.random() - 0.5) * 0.02
          particle.vy += (Math.random() - 0.5) * 0.02
        }

        // Update position
        particle.x += particle.vx
        particle.y += particle.vy

        // Boundary wrapping
        if (particle.x < 0) particle.x = canvas.width
        if (particle.x > canvas.width) particle.x = 0
        if (particle.y < 0) particle.y = canvas.height
        if (particle.y > canvas.height) particle.y = 0

        // Adjust brightness based on distance to mouse
        const brightnessFactor = distance < maxDistance ? 1 - distance / maxDistance : 0.3
        particle.brightness = 0.5 + brightnessFactor * 0.5

        // Draw particle with theme-aware colors
        const gradient = ctx.createRadialGradient(particle.x, particle.y, 0, particle.x, particle.y, particle.radius * 3)
        
        if (isDark) {
          // Dark mode: blue particles
          gradient.addColorStop(0, `rgba(0, 102, 204, ${0.8 * particle.brightness})`)
          gradient.addColorStop(0.7, `rgba(0, 102, 204, ${0.3 * particle.brightness})`)
          gradient.addColorStop(1, 'rgba(0, 102, 204, 0)')
        } else {
          // Light mode: charcoal particles
          gradient.addColorStop(0, `rgba(136, 136, 136, ${0.5 * particle.brightness})`)
          gradient.addColorStop(0.7, `rgba(136, 136, 136, ${0.2 * particle.brightness})`)
          gradient.addColorStop(1, 'rgba(136, 136, 136, 0)')
        }

        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.radius * 3, 0, Math.PI * 2)
        ctx.fill()

        // Core bright dot
        const coreColor = isDark ? 'rgba(0, 102, 204, ' : 'rgba(136, 136, 136, '
        ctx.fillStyle = coreColor + particle.brightness + ')'
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2)
        ctx.fill()
      })

      requestAnimationFrame(animate)
    }

    animate()

    // Handle window resize
    const handleResize = () => {
      setCanvasSize()
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10"
      style={{ display: 'block' }}
    />
  )
}
