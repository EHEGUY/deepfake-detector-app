'use client'

import { useEffect, useRef } from 'react'
import * as THREE from 'three'

export default function ParticleBackground() {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!mountRef.current) return

    // 1. SCENE SETUP (Centered Camera)
    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x000000, 0.001)

    // Camera is now perfectly centered at Y=0
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000)
    camera.position.set(0, 0, 800) 

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mountRef.current.appendChild(renderer.domElement)

    // 2. ULTRA-GLOW STAR TEXTURE
    const canvas = document.createElement('canvas')
    canvas.width = 32
    canvas.height = 32
    const context = canvas.getContext('2d')
    if (context) {
      const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16)
      gradient.addColorStop(0, 'rgba(255,255,255,1)')
      gradient.addColorStop(0.1, 'rgba(255,255,255,0.9)') // Hotter core
      gradient.addColorStop(0.4, 'rgba(255,255,255,0.2)') // Smoother falloff
      gradient.addColorStop(1, 'rgba(0,0,0,0)')
      context.fillStyle = gradient
      context.fillRect(0, 0, 32, 32)
    }
    const starTexture = new THREE.CanvasTexture(canvas)

    // 3. SPIRAL GALAXY MATH
    const particleCount = 12000
    const geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(particleCount * 3)
    const colors = new Float32Array(particleCount * 3)

    const colorWhite = new THREE.Color('#ffffff')
    const colorBlue = new THREE.Color('#0066cc')   // Détecteur Blue
    const colorPurple = new THREE.Color('#d946ef') // Neon Purple (Much more distinct)

    const branches = 4
    const spin = 1.5

    for (let i = 0; i < particleCount; i++) {
      const radius = Math.random() * 600
      const spinAngle = radius * (spin / 600)
      const branchAngle = ((i % branches) / branches) * Math.PI * 2
      
      const randomX = Math.pow(Math.random(), 3) * (Math.random() < 0.5 ? 1 : -1) * 80
      const randomY = Math.pow(Math.random(), 3) * (Math.random() < 0.5 ? 1 : -1) * 80
      const randomZ = Math.pow(Math.random(), 3) * (Math.random() < 0.5 ? 1 : -1) * 80

      positions[i * 3] = Math.cos(branchAngle + spinAngle) * radius + randomX
      positions[i * 3 + 1] = randomY 
      positions[i * 3 + 2] = Math.sin(branchAngle + spinAngle) * radius + randomZ

      // 4. COLOR DISTRIBUTION & BRIGHTNESS BOOST
      const colorMix = Math.random()
      const mixedColor = new THREE.Color()

      if (colorMix < 0.18) {
        // 18% Distinct Neon Purple
        mixedColor.copy(colorPurple) 
        mixedColor.multiplyScalar(1.5) // Boosts brightness so it pierces the blue
      } else if (colorMix < 0.55) {
        // 37% Deep Blue
        mixedColor.copy(colorBlue)   
        mixedColor.multiplyScalar(1.2)
      } else {
        // 45% White/Core
        mixedColor.copy(colorWhite)  
        mixedColor.multiplyScalar(0.8 + Math.random() * 0.4)
      }

      colors[i * 3] = mixedColor.r
      colors[i * 3 + 1] = mixedColor.g
      colors[i * 3 + 2] = mixedColor.b
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    // 5. LARGER, GLOWING MATERIAL
    const material = new THREE.PointsMaterial({
      size: 6.5, // Increased size for massive glow effect
      vertexColors: true,
      map: starTexture,
      transparent: true,
      opacity: 1.0, // Max opacity
      depthWrite: false, 
      blending: THREE.AdditiveBlending, 
    })

    const starSystem = new THREE.Points(geometry, material)
    
    // Tilt the galaxy forward so we look down into the spiral
    starSystem.rotation.x = 1.2 
    scene.add(starSystem)

    // 6. CENTERED INTERACTION LOGIC
    let mouseX = 0
    let mouseY = 0
    let targetX = 0
    let targetY = 0
    const windowHalfX = window.innerWidth / 2
    const windowHalfY = window.innerHeight / 2

    const onDocumentMouseMove = (event: MouseEvent) => {
      mouseX = (event.clientX - windowHalfX) / windowHalfX
      mouseY = (event.clientY - windowHalfY) / windowHalfY
    }
    window.addEventListener('mousemove', onDocumentMouseMove)

    // 7. ANIMATION LOOP
    const clock = new THREE.Clock()
    let animationFrameId: number

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate)
      const elapsedTime = clock.getElapsedTime()

      // The entire galaxy slowly rotates
      starSystem.rotation.z = elapsedTime * -0.05

      // FLUID CAMERA INTERACTION (Centered on Y=0)
      targetX = mouseX * 150 
      targetY = mouseY * 150 
      
      // The camera now orbits around absolute center (0,0)
      camera.position.x += (targetX - camera.position.x) * 0.02
      camera.position.y += (-targetY - camera.position.y) * 0.02
      
      camera.lookAt(0, 0, 0)

      renderer.render(scene, camera)
    }
    animate()

    // 8. RESIZE HANDLER
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      window.removeEventListener('mousemove', onDocumentMouseMove)
      cancelAnimationFrame(animationFrameId)
      if (mountRef.current) {
        mountRef.current.removeChild(renderer.domElement)
      }
      geometry.dispose()
      material.dispose()
      starTexture.dispose()
      renderer.dispose()
    }
  }, [])

  return (
    <div ref={mountRef} className="fixed inset-0 z-[-1] pointer-events-none bg-black" />
  )
}