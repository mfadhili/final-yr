import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { BookOpen, Lightbulb, Target, Zap } from "lucide-react"
import { Authors } from "@/components/authors"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="container mx-auto max-w-4xl">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">About the Algorithm</h1>
          <p className="text-xl text-gray-600">
            Understanding the Sobol-Halton Sequence based Hippopotamus Optimization (SHS-HO)
          </p>
        </div>

        {/* Introduction */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center">
              <BookOpen className="mr-2 h-5 w-5" />
              Introduction
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-gray-700">
              The Sobol-Halton Sequence Hippopotamus Optimization (SHS-HO) presents a novel metaheuristic approach for
              the optimal design and control of photovoltaic (PV) systems. This algorithm capitalizes on the strengths
              of Sobol and Halton quasi-random sequences to improve the results obtained using the Hippopotamus
              Optimization algorithm.
            </p>
            <p className="text-gray-700">
              The optimization of photovoltaic systems is essential for maximizing energy output, ensuring system
              efficiency, and maintaining system sustainability. SHS-HO addresses the critical challenges of premature
              convergence and non-diverse solution spaces that plague traditional metaheuristic algorithms.
            </p>
            <p className="text-gray-700">
              The sequences ensure diversity in the solution space and prevent premature convergence, making SHS-HO
              particularly effective for Maximum Power Point Tracking (MPPT) in PV systems under varying environmental
              conditions.
            </p>
          </CardContent>
        </Card>

        <Authors />

        {/* Key Components */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Target className="mr-2 h-5 w-5" />
              Key Components
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-lg mb-3 flex items-center">
                  <Zap className="mr-2 h-4 w-4 text-blue-600" />
                  Quasi-Random Sequences
                </h3>
                <ul className="space-y-2 text-gray-700">
                  <li>
                    • <strong>Sobol Sequences:</strong> Low-discrepancy sequences for uniform distribution
                  </li>
                  <li>
                    • <strong>Halton Sequences:</strong> Deterministic sampling for improved coverage
                  </li>
                  <li>
                    • <strong>Hybrid Approach:</strong> Combined benefits of both sequence types
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-3 flex items-center">
                  <Lightbulb className="mr-2 h-4 w-4 text-green-600" />
                  Hippopotamus Behavior
                </h3>
                <ul className="space-y-2 text-gray-700">
                  <li>
                    • <strong>Territorial Behavior:</strong> Local search intensification
                  </li>
                  <li>
                    • <strong>Social Hierarchy:</strong> Population-based exploration
                  </li>
                  <li>
                    • <strong>Adaptive Movement:</strong> Dynamic search strategy
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Algorithm Features */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Algorithm Features</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600 mb-2">Enhanced</div>
                <div className="text-sm text-gray-600">Convergence Speed</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600 mb-2">Improved</div>
                <div className="text-sm text-gray-600">Solution Quality</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600 mb-2">Balanced</div>
                <div className="text-sm text-gray-600">Exploration/Exploitation</div>
              </div>
            </div>

            <Separator className="my-6" />

            <div className="space-y-4">
              <h3 className="font-semibold text-lg">Technical Advantages</h3>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">Low Computational Complexity</Badge>
                <Badge variant="secondary">Parameter Self-Adaptation</Badge>
                <Badge variant="secondary">Premature Convergence Avoidance</Badge>
                <Badge variant="secondary">Multi-modal Optimization</Badge>
                <Badge variant="secondary">Scalable Architecture</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Mathematical Foundation */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Mathematical Foundation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-gray-100 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">PV System I-V Characteristic:</h4>
              <p className="font-mono text-sm mb-2">
                I(V;G,T) = I<sub>sc</sub> × [1 - exp(-V/V*<sub>oc</sub>)] × (G/1000)
              </p>
              <p className="font-mono text-sm">P(V;G,T) = V × I(V;G,T)</p>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">Quasi-Random Initialization:</h4>
              <p className="font-mono text-sm">
                x<sub>ij</sub> = l<sub>j</sub> + q<sub>ij</sub> × (h<sub>j</sub> - l<sub>j</sub>)
              </p>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">Fitness Function (Penalty Method):</h4>
              <p className="font-mono text-sm">
                Fitness(x) = f(x) + Σ λ<sub>i</sub> × penalty<sub>i</sub>(x)
              </p>
            </div>
            <p className="text-sm text-gray-600">
              Where G is irradiance (W/m²), T is temperature (°C), I<sub>sc</sub> = 10A, V<sub>oc</sub> = 100V, and q
              <sub>ij</sub> represents quasi-random sequences from Sobol or Halton generators.
            </p>
          </CardContent>
        </Card>

        {/* Applications */}
        <Card>
          <CardHeader>
            <CardTitle>PV System Applications</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold mb-3">Current Implementation</h3>
                <ul className="space-y-1 text-gray-700 text-sm">
                  <li>• Maximum Power Point Tracking (MPPT)</li>
                  <li>• Component sizing optimization</li>
                  <li>• System configuration design</li>
                  <li>• Grid integration efficiency</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-3">Future Applications</h3>
                <ul className="space-y-1 text-gray-700 text-sm">
                  <li>• Optimal site selection</li>
                  <li>• Solar tracking system optimization</li>
                  <li>• Inverter parameter tuning</li>
                  <li>• Energy management systems</li>
                  <li>• Panel cleaning and maintenance scheduling</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
