/**
 * TradeGlobe — self-contained 3D globe.
 * Supports external selectedEventId to fly from parent.
 */
import { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import Globe from 'react-globe.gl'
import { GeoEvent, GlobeArc, CATEGORY_COLORS } from '../types/globe'

const DIRECTION_COLORS: Record<string, string> = {
  bullish:  '#00e676',
  bearish:  '#ff1744',
  neutral:  '#00e5ff',
  risk:     '#ffd740',
}

interface TradeGlobeProps {
  events:          GeoEvent[]
  arcs:            GlobeArc[]
  focusLat:        number
  focusLng:        number
  focusAltitude?:  number
  onEventClick?:   (event: GeoEvent) => void
  selectedEventId?: string | null
}

export default function TradeGlobe({
  events, arcs, focusLat, focusLng, focusAltitude = 1.8,
  onEventClick, selectedEventId,
}: TradeGlobeProps) {
  const globeRef     = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hovered,   setHovered]   = useState<string | null>(null)
  const [selected,  setSelected]  = useState<string | null>(null)
  const [size,      setSize]      = useState({ w: 0, h: 0 })

  // Resize observer — also polls via rAF until non-zero dimensions are
  // found, guarding against the rare case where the parent visibility has
  // not yet propagated when the effect first runs.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    let rafId = 0

    const applySize = (w: number, h: number) => {
      if (w > 0 && h > 0) setSize({ w, h })
    }

    const ro = new ResizeObserver(entries => {
      for (const e of entries)
        applySize(e.contentRect.width, e.contentRect.height)
    })
    ro.observe(el)

    // Immediate read — works when parent is visible:hidden (layout intact)
    applySize(el.clientWidth, el.clientHeight)

    // Fallback: if still 0 after first frame (display:none ancestor), keep
    // retrying until we get real dimensions or the component unmounts.
    const poll = () => {
      const w = el.clientWidth, h = el.clientHeight
      if (w > 0 && h > 0) { setSize({ w, h }) }
      else rafId = requestAnimationFrame(poll)
    }
    if (el.clientWidth === 0) rafId = requestAnimationFrame(poll)

    return () => {
      ro.disconnect()
      cancelAnimationFrame(rafId)
    }
  }, [])

  // Globe controls
  useEffect(() => {
    const ctrl = globeRef.current?.controls()
    if (!ctrl) return
    ctrl.autoRotate      = true
    ctrl.autoRotateSpeed = 0.25
    ctrl.enableZoom      = true
    ctrl.minDistance     = 140
    ctrl.maxDistance     = 700
  }, [])

  // Fly to focus on mount
  useEffect(() => {
    if (!globeRef.current || !focusLat) return
    const timer = setTimeout(() => {
      const ctrl = globeRef.current?.controls()
      if (ctrl) ctrl.autoRotate = false
      globeRef.current?.pointOfView({ lat: focusLat, lng: focusLng, altitude: focusAltitude }, 1600)
    }, 700)
    return () => clearTimeout(timer)
  }, [focusLat, focusLng, focusAltitude])

  // External selection → fly
  useEffect(() => {
    if (selectedEventId === undefined) return
    setSelected(selectedEventId)
    if (!selectedEventId || !globeRef.current) return
    const ev = events.find(e => e.id === selectedEventId)
    if (!ev) return
    const ctrl = globeRef.current.controls()
    if (ctrl) ctrl.autoRotate = false
    globeRef.current.pointOfView({ lat: ev.lat, lng: ev.lng, altitude: 1.5 }, 1000)
  }, [selectedEventId, events])

  // Internal selection → fly
  useEffect(() => {
    if (!selected || !globeRef.current) return
    const ev = events.find(e => e.id === selected)
    if (!ev) return
    const ctrl = globeRef.current.controls()
    if (ctrl) ctrl.autoRotate = false
    globeRef.current.pointOfView({ lat: ev.lat, lng: ev.lng, altitude: 1.4 }, 900)
  }, [selected, events])

  const pointColor = useCallback((d: any) => {
    if (d.id === selected) return '#ffffff'
    if (d.id === hovered)  return '#ffffff'
    if (d.direction_color) return d.direction_color
    if (d.direction)       return DIRECTION_COLORS[d.direction] ?? '#00e5ff'
    return CATEGORY_COLORS[d.category as keyof typeof CATEGORY_COLORS] ?? '#00e5ff'
  }, [selected, hovered])

  const pointRadius = useCallback((d: any) => {
    const base = 0.20 + d.severity * 0.20
    return d.id === hovered ? base * 1.6 : d.id === selected ? base * 1.3 : base
  }, [hovered, selected])

  const arcColor = useCallback((d: any) => {
    const col = d.direction_color
      ?? DIRECTION_COLORS[d.direction ?? '']
      ?? CATEGORY_COLORS[d.category as keyof typeof CATEGORY_COLORS]
      ?? '#00e5ff'
    return [`${col}10`, `${col}bb`]
  }, [])

  const arcStroke  = useCallback((d: any) => d.strength * 1.4, [])
  const arcAlt     = useCallback((d: any) => 0.10 + d.strength * 0.25, [])

  const tooltipData = useMemo(
    () => events.filter(e => e.id === hovered),
    [events, hovered],
  )

  const htmlElement = useCallback((d: any) => {
    const el = document.createElement('div')
    el.style.cssText = [
      'background:rgba(3,8,14,0.94)',
      'border:1px solid rgba(255,255,255,0.12)',
      'border-radius:2px',
      'padding:7px 11px',
      'color:#ccd8e8',
      'font-size:11px',
      'max-width:240px',
      'pointer-events:none',
      'white-space:pre-wrap',
      'line-height:1.4',
      'font-family:Inter,sans-serif',
    ].join(';')
    el.textContent = d.title
    return el
  }, [])

  return (
    <div ref={containerRef} className="w-full h-full" style={{ background: '#000508' }}>
      {size.w > 0 && (
        <Globe
          ref={globeRef}
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
          backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
          atmosphereColor="#003060"
          atmosphereAltitude={0.18}
          width={size.w}
          height={size.h}

          pointsData={events}
          pointLat={(d: any) => d.lat}
          pointLng={(d: any) => d.lng}
          pointColor={pointColor}
          pointAltitude={0.014}
          pointRadius={pointRadius}
          pointsMerge={false}
          onPointClick={(point: any) => {
            const id = point.id === selected ? null : point.id
            setSelected(id)
            if (id && onEventClick) {
              const ev = events.find(e => e.id === id)
              if (ev) onEventClick(ev)
            }
          }}
          onPointHover={(point: any) => setHovered(point?.id ?? null)}

          arcsData={arcs}
          arcStartLat={(d: any) => d.source_lat}
          arcStartLng={(d: any) => d.source_lng}
          arcEndLat={(d: any) => d.target_lat}
          arcEndLng={(d: any) => d.target_lng}
          arcColor={arcColor}
          arcAltitude={arcAlt}
          arcStroke={arcStroke}
          arcDashLength={0.4}
          arcDashGap={0.18}
          arcDashAnimateTime={2200}

          htmlElementsData={tooltipData}
          htmlElement={htmlElement}
          htmlAltitude={0.06}
        />
      )}
    </div>
  )
}
