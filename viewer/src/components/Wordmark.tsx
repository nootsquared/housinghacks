import { Link } from "react-router-dom"

export function Wordmark() {
  return (
    <Link to="/" className="inline-flex items-center gap-2 text-[15px] font-semibold tracking-tight">
      <svg viewBox="0 0 20 20" className="size-5" aria-hidden>
        <path d="M3 9.2 10 3l7 6.2V17H3z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
        <path d="M8 17v-4.5h4V17" fill="var(--brand)" />
      </svg>
      Openhouse
    </Link>
  )
}
