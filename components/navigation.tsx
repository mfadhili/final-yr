"use client"

import Link from "next/link"
import Image from "next/image"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Home, BookOpen, Play, BarChart3 } from "lucide-react"

const navigation = [
  { name: "Home", href: "/", icon: Home },
  { name: "About", href: "/about", icon: BookOpen },
  { name: "Simulation", href: "/simulation", icon: Play },
  { name: "Results", href: "/results", icon: BarChart3 },
]

export function Navigation() {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center">
        <div className="mr-4 flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <div className="h-10 w-10 relative">
              <Image src="/images/uon-logo.png" alt="University of Nairobi Logo" fill className="object-contain" />
            </div>
            <span className="hidden font-bold sm:inline-block">SHS-HO Research</span>
          </Link>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center space-x-2 transition-colors hover:text-foreground/80",
                    pathname === item.href ? "text-foreground" : "text-foreground/60",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.name}</span>
                </Link>
              )
            })}
          </nav>
        </div>
        <div className="ml-auto flex items-center space-x-4">
          <Button variant="outline" size="sm" asChild>
            <Link href="/simulation">
              <Play className="mr-2 h-4 w-4" />
              Run Simulation
            </Link>
          </Button>
        </div>
      </div>
    </header>
  )
}
