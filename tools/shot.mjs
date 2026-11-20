import puppeteer from "puppeteer-core"

const [, , url, out = "shot.png", ...steps] = process.argv
const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: "new",
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
  defaultViewport: { width: 1440, height: 900, deviceScaleFactor: 1 },
})
const page = await browser.newPage()
const logs = []
page.on("console", (m) => ["error", "warn"].includes(m.type()) && logs.push(`${m.type()}: ${m.text()}`))
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`))
await page.goto(url, { waitUntil: "networkidle0", timeout: 60000 })
await new Promise((r) => setTimeout(r, 2500))
let n = 0
for (const step of steps) {
  const [kind, arg] = step.split("=")
  if (kind === "click") {
    await page.evaluate((t) => {
      const el = [...document.querySelectorAll("button, a")].find((b) => b.textContent.trim() === t || b.getAttribute("aria-label") === t)
      el?.click()
    }, arg)
  } else if (kind === "thumb") {
    await page.evaluate((i) => document.querySelectorAll("main button img")[+i]?.parentElement.click(), arg)
  } else if (kind === "key") {
    await page.keyboard.press(arg)
  } else if (kind === "wait") {
    await new Promise((r) => setTimeout(r, +arg))
  } else if (kind === "shot") {
    await page.screenshot({ path: out.replace(/\.png$/, `-${n++}.png`) })
  } else if (kind === "eval") {
    logs.push(`eval: ${JSON.stringify(await page.evaluate(arg))}`)
  }
}
await page.screenshot({ path: out })
console.log(logs.filter((l) => !l.includes("THREE.Clock")).join("\n") || "no errors")
await browser.close()
