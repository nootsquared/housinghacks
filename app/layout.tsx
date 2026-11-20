import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"

const sans = Geist({ subsets: ["latin"], variable: "--font-geist", display: "swap" })
const mono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono", display: "swap" })

export const metadata: Metadata = {
  title: "Openhouse · Ann Arbor rentals",
  description: "Every Ann Arbor rental we could find, on a satellite map with real building outlines and a noise model.",
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className={`${sans.variable} ${mono.variable} antialiased`}>{children}</body>
    </html>
  )
}
