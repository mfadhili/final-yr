"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Play, Square, RotateCcw, Settings, TrendingUp } from "lucide-react"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js"
import { Line } from "react-chartjs-2"

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const algorithms = [
  { value: "Hippopotamus", label: "Hippopotamus", color: "rgb(59, 130, 246)" },
  { value: "PSO", label: "PSO", color: "rgb(147, 51, 234)" },
  { value: "ABC", label: "ABC", color: "rgb(139, 69, 19)" },
  { value: "SA", label: "SA", color: "rgb(107, 114, 128)" },
  { value: "GWO", label: "GWO", color: "rgb(219, 39, 119)" },
  { value: "HS", label: "HS", color: "rgb(6, 182, 212)" },
  { value: "CSA", label: "CSA", color: "rgb(0, 0, 0)" },
  { value: "LSA", label: "LSA", color: "rgb(249, 115, 22)" },
  { value: "EPO", label: "EPO", color: "rgb(132, 204, 22)" },
  { value: "Cuckoo Search", label: "Cuckoo Search", color: "rgb(255, 215, 0)" },
  { value: "ABO", label: "ABO", color: "rgb(255, 140, 0)" },
  { value: "HEM", label: "HEM", color: "rgb(25, 25, 112)" },
  { value: "WSO", label: "WSO", color: "rgb(34, 139, 34)" },
  { value: "DE", label: "DE", color: "rgb(139, 69, 19)" },
  { value: "TLBO", label: "TLBO", color: "rgb(220, 20, 60)" },
  { value: "Genetic Alg", label: "Genetic Alg", color: "rgb(0, 128, 0)" },
]

export default function SimulationPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [iterations, setIterations] = useState([100])
  const [populationSize, setPopulationSize] = useState([30])
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>(["Hippopotamus", "PSO", "ABC"])
  const [simulationResults, setSimulationResults] = useState<any>(null)
  const [animationProgress, setAnimationProgress] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const animationRef = useRef<NodeJS.Timeout>()

  const handleStartSimulation = async () => {
    if (selectedAlgorithms.length === 0) return

    setIsRunning(true)
    setProgress(0)
    setSimulationResults(null)

    try {
      const response = await fetch("/api/run-simulation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pop_size: populationSize[0],
          max_iter: iterations[0],
        }),
      })

      if (!response.ok) {
        throw new Error("Simulation failed")
      }

      const data = await response.json()
      setSimulationResults(data.results)
      setProgress(100)

      // Start animation
      startAnimation()
    } catch (error) {
      console.error("Simulation error:", error)
      alert("Simulation failed. Please try again.")
    } finally {
      setIsRunning(false)
    }
  }

  const startAnimation = () => {
    setIsAnimating(true)
    setAnimationProgress(0)

    const totalDuration = 7000 // 7 seconds
    const intervalTime = 50 // 50ms intervals
    const totalSteps = totalDuration / intervalTime
    const stepSize = iterations[0] / totalSteps

    let currentStep = 0

    animationRef.current = setInterval(() => {
      currentStep += stepSize
      if (currentStep >= iterations[0]) {
        currentStep = iterations[0]
        setIsAnimating(false)
        if (animationRef.current) {
          clearInterval(animationRef.current)
        }
      }
      setAnimationProgress(Math.floor(currentStep))
    }, intervalTime)
  }

  const handleStopSimulation = () => {
    setIsRunning(false)
    setProgress(0)
    if (animationRef.current) {
      clearInterval(animationRef.current)
    }
    setIsAnimating(false)
  }

  const handleReset = () => {
    setIsRunning(false)
    setProgress(0)
    setSimulationResults(null)
    setAnimationProgress(0)
    setIsAnimating(false)
    if (animationRef.current) {
      clearInterval(animationRef.current)
    }
  }

  const getChartData = () => {
    if (!simulationResults || simulationResults.length === 0) return null

    const maxIteration = isAnimating ? animationProgress : iterations[0]
    const filteredData = simulationResults.slice(0, maxIteration)

    const datasets = selectedAlgorithms.map((algorithm) => {
      const algorithmData = algorithms.find((a) => a.value === algorithm)
      return {
        label: algorithm,
        data: filteredData.map((row: any) => Number.parseFloat(row[algorithm]) || 0),
        borderColor: algorithmData?.color || "rgb(75, 192, 192)",
        backgroundColor: algorithmData?.color || "rgb(75, 192, 192)",
        tension: 0.1,
        pointRadius: 2,
        pointHoverRadius: 4,
      }
    })

    return {
      labels: filteredData.map((row: any) => row.Iteration),
      datasets,
    }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: true,
        text: "Power Output Convergence",
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Iteration",
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Power Output (W)",
        },
      },
    },
    animation: {
      duration: 0, // Disable chart.js animation since we're controlling it manually
    },
  }

  const toggleAlgorithm = (algorithm: string) => {
    setSelectedAlgorithms((prev) =>
      prev.includes(algorithm) ? prev.filter((a) => a !== algorithm) : [...prev, algorithm],
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Algorithm Simulation</h1>
          <p className="text-xl text-gray-600">
            Configure and run the Sobol-Halton Sequence Hippopotamus Optimization (SHS-HO)
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Configuration Panel */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="mr-2 h-5 w-5" />
                  Configuration
                </CardTitle>
                <CardDescription>Set algorithm parameters and select algorithms to compare</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Population Size: {populationSize[0]}</Label>
                  <Slider
                    value={populationSize}
                    onValueChange={setPopulationSize}
                    max={100}
                    min={10}
                    step={5}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Max Iterations: {iterations[0]}</Label>
                  <Slider
                    value={iterations}
                    onValueChange={setIterations}
                    max={1000}
                    min={50}
                    step={50}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Select Algorithms to Compare:</Label>
                  <div className="max-h-48 overflow-y-auto space-y-2 border rounded-md p-2">
                    {algorithms.map((algorithm) => (
                      <div key={algorithm.value} className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id={algorithm.value}
                          checked={selectedAlgorithms.includes(algorithm.value)}
                          onChange={() => toggleAlgorithm(algorithm.value)}
                          className="rounded"
                        />
                        <label htmlFor={algorithm.value} className="text-sm flex items-center space-x-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: algorithm.color }} />
                          <span>{algorithm.label}</span>
                        </label>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    onClick={handleStartSimulation}
                    disabled={isRunning || selectedAlgorithms.length === 0}
                    className="flex-1"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    Start
                  </Button>
                  <Button onClick={handleStopSimulation} disabled={!isRunning && !isAnimating} variant="outline">
                    <Square className="mr-2 h-4 w-4" />
                    Stop
                  </Button>
                  <Button onClick={handleReset} variant="outline">
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Simulation Display */}
          <div className="lg:col-span-2 space-y-6">
            {/* Status Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center">
                    <TrendingUp className="mr-2 h-5 w-5" />
                    Simulation Status
                  </span>
                  <Badge variant={isRunning ? "default" : isAnimating ? "secondary" : "outline"}>
                    {isRunning ? "Running" : isAnimating ? "Animating" : "Idle"}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{animationProgress}</div>
                    <div className="text-sm text-gray-600">Current Iteration</div>
                  </div>
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">{iterations[0]}</div>
                    <div className="text-sm text-gray-600">Max Iterations</div>
                  </div>
                  <div className="text-center p-3 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">{populationSize[0]}</div>
                    <div className="text-sm text-gray-600">Population Size</div>
                  </div>
                  <div className="text-center p-3 bg-orange-50 rounded-lg">
                    <div className="text-2xl font-bold text-orange-600">{selectedAlgorithms.length}</div>
                    <div className="text-sm text-gray-600">Algorithms</div>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              </CardContent>
            </Card>

            {/* Visualization */}
            <Card>
              <CardHeader>
                <CardTitle>Convergence Visualization</CardTitle>
                <CardDescription>Real-time convergence curve showing power output over iterations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96 w-full">
                  {simulationResults && getChartData() ? (
                    <Line data={getChartData()!} options={chartOptions} />
                  ) : (
                    <div className="h-full bg-gray-100 rounded-lg flex items-center justify-center">
                      <div className="text-center text-gray-500">
                        <TrendingUp className="h-12 w-12 mx-auto mb-2 opacity-50" />
                        <p>Convergence chart will appear here during simulation</p>
                        <p className="text-sm">Select algorithms and start simulation to view</p>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Algorithm Info */}
            <Card>
              <CardHeader>
                <CardTitle>Algorithm Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <h4 className="font-semibold mb-2">Current Configuration:</h4>
                    <ul className="space-y-1 text-gray-600">
                      <li>Population Size: {populationSize[0]} individuals</li>
                      <li>Max Iterations: {iterations[0]}</li>
                      <li>Selected Algorithms: {selectedAlgorithms.length}</li>
                      <li>Irradiance: 800 W/m²</li>
                      <li>Temperature: 25°C</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Selected Algorithms:</h4>
                    <div className="flex flex-wrap gap-1">
                      {selectedAlgorithms.map((algorithm) => (
                        <Badge key={algorithm} variant="secondary" className="text-xs">
                          {algorithm}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
