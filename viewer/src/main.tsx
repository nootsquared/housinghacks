import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter, Route, Routes } from "react-router-dom"
import { TooltipProvider } from "@/components/ui/tooltip"
import Explore from "@/pages/Explore"
import ListingPage from "@/pages/Listing"
import "maplibre-gl/dist/maplibre-gl.css"
import "./index.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <TooltipProvider delayDuration={200}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Explore />} />
          <Route path="/l/:id" element={<ListingPage />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </StrictMode>,
)
