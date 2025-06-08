import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { Navigation } from "@/components/navigation"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "SHS-HO Research | Sobol-Halton Sequence Hippopotamus Optimization",
  description:
    "Research presentation on the Sobol-Halton Sequence Hippopotamus Optimization (SHS-HO) - an innovative hybrid optimization approach for photovoltaic systems.",
  icons: {
    icon: "/images/uon-logo.png",
    apple: "/images/uon-logo.png",
  },
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Navigation />
        <main>{children}</main>
        <footer className="border-t py-6 md:py-0">
          <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
            <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
              <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
                © 2025 SHS-HO Research Project.
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  )
}
