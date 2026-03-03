'use client'

import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

export const Background3D = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const animationFrameRef = useRef<number>()
  const cubeGroupRef = useRef<THREE.Group | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return

    // Scene setup
    const scene = new THREE.Scene()
    scene.background = null
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    )
    camera.position.set(0, 0, 5)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    })
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000000, 0)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    rendererRef.current = renderer

    // Cube group
    const cubeGroup = new THREE.Group()
    scene.add(cubeGroup)
    cubeGroupRef.current = cubeGroup

    const geometry = new THREE.BoxGeometry(2, 2, 2, 4, 4, 4)

    const vertexShader = `
      varying vec2 vUv;
      varying vec3 vNormal;
      void main() {
        vUv = uv;
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `

    const fragmentShader = `
      uniform float iTime;
      varying vec2 vUv;
      varying vec3 vNormal;
      
      void main() {
        vec3 color = mix(vec3(0.1, 0.1, 0.15), vec3(0.2, 0.15, 0.25), vUv.x);
        float edge = 1.0 - max(abs(vUv.x - 0.5), abs(vUv.y - 0.5)) * 2.0;
        edge = pow(edge, 4.0);
        color += edge * vec3(1.0, 0.2, 0.2) * 0.6;
        gl_FragColor = vec4(color, 1.0);
      }
    `

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        iTime: { value: 0 }
      },
      transparent: true,
      side: THREE.DoubleSide
    })

    const cube = new THREE.Mesh(geometry, material)
    cubeGroup.add(cube)

    const wireframe = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, 10),
      new THREE.LineBasicMaterial({
        color: 0xff3333,
        linewidth: 1.5,
        transparent: true,
        opacity: 0.15
      })
    )
    wireframe.scale.setScalar(1.001)
    cubeGroup.add(wireframe)

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5)
    directionalLight.position.set(5, 10, 7)
    scene.add(directionalLight)

    const pointLight = new THREE.PointLight(0xff3333, 1, 20)
    pointLight.position.set(-3, 2, 5)
    scene.add(pointLight)

    // Mouse interaction
    const mouse = new THREE.Vector2(0, 0)

    const handleMouseMove = (event: MouseEvent) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1

      if (!ScrollTrigger.isScrolling()) {
        gsap.to(cubeGroup.rotation, {
          x: '+=' + (mouse.y * 0.03 - cubeGroup.rotation.x * 0.02),
          y: '+=' + (mouse.x * 0.03 - cubeGroup.rotation.y * 0.02),
          duration: 1,
          ease: 'power2.out',
          overwrite: 'auto'
        })
      }
    }

    window.addEventListener('mousemove', handleMouseMove)

    // Scroll animation
    const scrollTimeline = gsap.timeline({
      scrollTrigger: {
        trigger: '.content',
        start: 'top top',
        end: 'bottom bottom',
        scrub: 1.5,
        onUpdate: (self) => {
          const progress = self.progress
          const minFOV = 20
          const maxFOV = 60
          const zoomCurve = progress < 0.5 ? progress * 2 : 2 - progress * 2
          camera.fov = maxFOV - (maxFOV - minFOV) * zoomCurve
          camera.updateProjectionMatrix()
        }
      }
    })

    scrollTimeline
      .to(
        cubeGroup.rotation,
        {
          x: Math.PI * 1.2,
          y: Math.PI * 2,
          z: Math.PI * 0.3,
          ease: 'power2.inOut',
          immediateRender: false
        },
        0
      )
      .to(
        camera.position,
        {
          z: 0.8,
          y: 0.2,
          ease: 'power2.inOut'
        },
        0.5
      )
      .to(
        camera.position,
        {
          z: 4.0,
          y: 0,
          ease: 'power2.inOut'
        },
        1.0
      )

    // Animation loop
    function animate() {
      animationFrameRef.current = requestAnimationFrame(animate)
      cubeGroup.rotation.x += 0.0005
      cubeGroup.rotation.y += 0.0008
      renderer.render(scene, camera)
    }

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
    }

    window.addEventListener('resize', handleResize)
    animate()

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('resize', handleResize)
      ScrollTrigger.getAll().forEach((trigger) => trigger.kill())
      rendererRef.current?.dispose()
      sceneRef.current?.clear()
    }
  }, [])

  return <canvas ref={canvasRef} className="fixed top-0 left-0 w-full h-full z-0 pointer-events-none" />
}
