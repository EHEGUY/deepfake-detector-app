import { useEffect } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

export function useScrollAnimations() {
  useEffect(() => {
    const sections = document.querySelectorAll('.section')

    sections.forEach((section) => {
      const title = section.querySelector('.title')
      const description = section.querySelector('.description')

      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: section,
          start: 'top 80%',
          end: 'top 20%',
          scrub: 1,
          toggleActions: 'play none none reverse'
        }
      })

      if (title) {
        tl.to(
          title,
          {
            opacity: 1,
            y: 0,
            duration: 1,
            ease: 'power2.out'
          },
          0
        )
      }

      if (description) {
        tl.to(
          description,
          {
            opacity: 1,
            y: 0,
            duration: 1,
            ease: 'power2.out',
            delay: 0.2
          },
          0
        )
      }
    })

    return () => {
      ScrollTrigger.getAll().forEach((trigger) => trigger.kill())
    }
  }, [])
}
