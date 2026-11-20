import { useEffect, useState } from "react"
import type { Listing, Scene } from "./types"

const cache = new Map<string, Promise<unknown>>()

function load<T>(url: string): Promise<T> {
  if (!cache.has(url)) {
    cache.set(
      url,
      fetch(url).then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${url}`)
        return r.json()
      }),
    )
  }
  return cache.get(url) as Promise<T>
}

function useJson<T>(url: string | null) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  useEffect(() => {
    if (!url) return
    let live = true
    load<T>(url).then(
      (d) => live && setData(d),
      (e) => live && setError(e),
    )
    return () => {
      live = false
    }
  }, [url])
  return { data, error }
}

export const asset = (id: string, file: string) => `/data/${id}/${file}`

export const useListings = () => useJson<Listing[]>("/data/index.json")

export function useListing(id: string | undefined) {
  const { data, error } = useListings()
  return { listing: data?.find((l) => l.id === id) ?? null, loading: !data && !error }
}

export const useScene = (id: string | undefined, enabled: boolean) =>
  useJson<Scene>(id && enabled ? `/data/${id}/scene.json` : null)

export const money = (n: number | null) =>
  n == null ? "—" : n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 })
