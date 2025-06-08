import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, BarChart3, BookOpen, Play } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Sobol-Halton Sequence
            <span className="block text-indigo-600">Hippopotamus Optimization (SHS-HO)</span>
            <span className="block text-2xl font-normal text-gray-600 mt-2">for Photovoltaic Systems</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            A novel metaheuristic approach for optimal design and control of PV systems, combining quasi-random
            sequences with bio-inspired hippopotamus behavior to maximize energy output and ensure system efficiency.
          </p>
          <div className="flex gap-4 justify-center">
            <Button asChild size="lg">
              <Link href="/simulation">
                <Play className="mr-2 h-5 w-5" />
                Run MPPT Simulation
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link href="/about">
                <BookOpen className="mr-2 h-5 w-5" />
                Learn More
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 bg-white">
        <div className="container mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center">
              <CardHeader>
                <div className="mx-auto w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                  <BarChart3 className="h-6 w-6 text-blue-600" />
                </div>
                <CardTitle>Maximum Power Point Tracking</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Optimizes voltage tracking for maximum power extraction from PV panels under varying irradiance and
                  temperature conditions, achieving superior convergence in 76 iterations.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <div className="mx-auto w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                  <Play className="h-6 w-6 text-green-600" />
                </div>
                <CardTitle>Quasi-Random Initialization</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Utilizes Sobol and Halton sequences for uniform population distribution, preventing premature
                  convergence and ensuring comprehensive solution space exploration.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <div className="mx-auto w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                  <ArrowRight className="h-6 w-6 text-purple-600" />
                </div>
                <CardTitle>Superior Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Achieves 846.29W maximum power output, outperforming 16 other metaheuristic algorithms including PSO,
                  GA, and DE in photovoltaic system optimization.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Quick Navigation */}
      <section className="py-16 px-4 bg-gray-50">
        <div className="container mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Explore the Research</h2>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BookOpen className="mr-2 h-5 w-5" />
                  About the Algorithm
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="mb-4">
                  Learn about the theoretical foundation, mathematical formulation, and innovative aspects of our hybrid
                  approach.
                </CardDescription>
                <Button variant="outline" asChild>
                  <Link href="/about">
                    Read More <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="mr-2 h-5 w-5" />
                  View Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="mb-4">
                  Explore comprehensive performance analysis, benchmark comparisons, and statistical validation of the
                  algorithm.
                </CardDescription>
                <Button variant="outline" asChild>
                  <Link href="/results">
                    View Results <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
    </div>
  )
}
