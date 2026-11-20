import { readFile } from "node:fs/promises"
import path from "node:path"

export const runtime = "nodejs"

const ROOT = path.resolve(process.cwd(), "../web/public/data")

const TYPES: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
}

export async function GET(_req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  const parts = (await ctx.params).path
  if (!parts?.length || parts.some((part) => part === ".." || part.includes("\0") || part.includes("/"))) {
    return new Response("Not found", { status: 404 })
  }
  const file = path.resolve(ROOT, ...parts)
  if (file !== ROOT && !file.startsWith(ROOT + path.sep)) return new Response("Not found", { status: 404 })
  const type = TYPES[path.extname(file).toLowerCase()]
  if (!type) return new Response("Not found", { status: 404 })
  try {
    const data = await readFile(file)
    return new Response(data, {
      headers: { "content-type": type, "cache-control": "public, max-age=86400" },
    })
  } catch {
    return new Response("Not found", { status: 404 })
  }
}
