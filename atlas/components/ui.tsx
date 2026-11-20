"use client"

import { useEffect, useRef, useState, type ReactNode } from "react"
import { cn } from "@/lib/cn"

export function Popover({
  label,
  active,
  children,
  align = "left",
  width = 300,
}: {
  label: ReactNode
  active?: boolean
  children: ReactNode
  align?: "left" | "right"
  width?: number
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false)
    }
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false)
    window.addEventListener("mousedown", onDown)
    window.addEventListener("keydown", onKey)
    return () => {
      window.removeEventListener("mousedown", onDown)
      window.removeEventListener("keydown", onKey)
    }
  }, [open])

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className={cn(
          "flex h-8 items-center gap-1.5 rounded-md border px-3 text-[13px] whitespace-nowrap transition-colors",
          active ? "border-text/40 bg-raised text-text" : "border-line-strong text-muted hover:bg-hover hover:text-text",
        )}
      >
        {label}
        <svg width="10" height="10" viewBox="0 0 10 10" className={cn("transition-transform", open && "rotate-180")}>
          <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.3" />
        </svg>
      </button>
      {open && (
        <div
          className={cn(
            "fade-in absolute top-10 z-50 rounded-lg border border-line-strong bg-raised p-4 shadow-[0_24px_60px_rgba(0,0,0,0.55)]",
            align === "right" ? "right-0" : "left-0",
          )}
          style={{ width }}
        >
          {children}
        </div>
      )}
    </div>
  )
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  size = "md",
}: {
  value: T
  options: { id: T; label: ReactNode }[]
  onChange: (value: T) => void
  size?: "sm" | "md"
}) {
  return (
    <div className="flex rounded-md border border-line-strong bg-bg p-0.5">
      {options.map((option) => (
        <button
          key={String(option.id)}
          type="button"
          aria-pressed={value === option.id}
          onClick={() => onChange(option.id)}
          className={cn(
            "min-w-0 flex-1 rounded-[5px] px-2 whitespace-nowrap transition-colors",
            size === "sm" ? "h-6 text-[12px]" : "h-7 text-[13px]",
            value === option.id ? "bg-text font-medium text-bg" : "text-muted hover:text-text",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

export function Check({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 py-1.5 text-[13px]">
      <span>{children}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn("relative h-[18px] w-8 shrink-0 rounded-full transition-colors", checked ? "bg-text" : "bg-line-strong")}
      >
        <span
          className={cn(
            "absolute top-[2px] size-[14px] rounded-full transition-all",
            checked ? "left-[16px] bg-bg" : "left-[2px] bg-muted",
          )}
        />
      </button>
    </label>
  )
}

export function Section({ title, aside, children, className }: { title: string; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("rounded-xl border border-line bg-surface", className)}>
      <header className="flex items-center justify-between gap-3 border-b border-line px-5 py-3">
        <h3 className="text-[13px] font-medium">{title}</h3>
        {aside}
      </header>
      <div className="p-5">{children}</div>
    </section>
  )
}

export function Tag({ children, tone }: { children: ReactNode; tone?: "accent" | "good" | "muted" }) {
  return (
    <span
      className={cn(
        "inline-flex h-[22px] items-center rounded-[5px] border px-1.5 text-[11.5px] whitespace-nowrap",
        tone === "accent" && "border-accent/30 bg-accent/10 text-accent",
        tone === "good" && "border-emerald-400/25 bg-emerald-400/10 text-emerald-300",
        (!tone || tone === "muted") && "border-line-strong bg-white/[0.03] text-muted",
      )}
    >
      {children}
    </span>
  )
}
