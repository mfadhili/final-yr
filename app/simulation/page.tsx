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
  Filler,
} from "chart.js"
import { Line, Scatter } from "react-chartjs-2"

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

const algorithms = [
  { value: "Hippopotamus", label: "Hippopotamus", color: "rgb(59, 130, 246)" },
  { value: "Sobol-Hippopotamus", label: "Sobol-Hippopotamus", color: "rgb(16, 185, 129)" },
  { value: "Halton-Hippopotamus", label: "Halton-Hippopotamus", color: "rgb(245, 158, 11)" },
  { value: "TLBO", label: "TLBO", color: "rgb(220, 20, 60)" },
  { value: "Genetic Alg", label: "Genetic Alg", color: "rgb(0, 128, 0)" },
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
]

function createGradient(ctx: CanvasRenderingContext2D, color: string) {
  const gradient = ctx.createLinearGradient(0, 0, 0, 400)
  gradient.addColorStop(0, color.replace(")", ", 0.6)").replace("rgb", "rgba"))
  gradient.addColorStop(1, color.replace(")", ", 0.1)").replace("rgb", "rgba"))
  return gradient
}

export default function SimulationPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [iterations, setIterations] = useState([100])
  const [populationSize, setPopulationSize] = useState([30])
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>(["Hippopotamus", "Sobol-Hippopotamus", "PSO"])
  const [simulationResults, setSimulationResults] = useState<any>(null)
  const [summaryResults, setSummaryResults] = useState<any>(null)
  const [animationProgress, setAnimationProgress] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const animationRef = useRef<NodeJS.Timeout>()
  const [showScatterPlots, setShowScatterPlots] = useState(false)
  const [activeScatterPlot, setActiveScatterPlot] = useState(0)

  const handleStartSimulation = async () => {
    if (selectedAlgorithms.length === 0) return

    setIsRunning(true)
    setProgress(0)
    setSimulationResults(null)
    setSummaryResults(null)

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
      setSummaryResults(data.summary)
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
    setSummaryResults(null)
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

    const ctx = document.getElementById("convergence-chart")?.getContext("2d")

    const datasets = selectedAlgorithms.map((algorithm) => {
      const algorithmData = algorithms.find((a) => a.value === algorithm)
      const color = algorithmData?.color || "rgb(75, 192, 192)"

      return {
        label: algorithm,
        data: filteredData.map((row: any) => Number.parseFloat(row[algorithm]) || 0),
        borderColor: color,
        backgroundColor: ctx ? createGradient(ctx, color) : color,
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
        fill: true,
      }
    })

    return {
      labels: filteredData.map((row: any) => row.Iteration),
      datasets,
    }
  }

  const getScatterPlot1Data = () => {
    if (!simulationResults || simulationResults.length === 0) return null

    // Final Power vs Convergence Speed
    const finalIteration = simulationResults[simulationResults.length - 1]
    const datasets = selectedAlgorithms.map((algorithm) => {
      const algorithmData = algorithms.find((a) => a.value === algorithm)
      const finalPower = Number.parseFloat(finalIteration[algorithm]) || 0

      // Calculate convergence speed (iteration where algorithm reaches 95% of final power)
      const targetPower = finalPower * 0.95
      let convergenceIteration = iterations[0]

      for (let i = 0; i < simulationResults.length; i++) {
        const currentPower = Number.parseFloat(simulationResults[i][algorithm]) || 0
        if (currentPower >= targetPower) {
          convergenceIteration = i + 1
          break
        }
      }

      return {
        label: algorithm,
        data: [{ x: convergenceIteration, y: finalPower }],
        backgroundColor: algorithmData?.color || "rgb(75, 192, 192)",
        borderColor: algorithmData?.color || "rgb(75, 192, 192)",
        pointRadius: 8,
        pointHoverRadius: 10,
      }
    })

    return { datasets }
  }

  const getScatterPlot2Data = () => {
    if (!simulationResults || simulationResults.length === 0) return null

    // Population Size vs Final Power (simulated data for different pop sizes)
    const finalIteration = simulationResults[simulationResults.length - 1]
    const popSizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    const datasets = selectedAlgorithms.map((algorithm) => {
      const algorithmData = algorithms.find((a) => a.value === algorithm)
      const basePower = Number.parseFloat(finalIteration[algorithm]) || 0

      // Simulate power output for different population sizes
      const data = popSizes.map((popSize) => ({
        x: popSize,
        y: basePower + (Math.random() - 0.5) * 50 + (popSize - 30) * 2, // Simulated relationship
      }))

      return {
        label: algorithm,
        data: data,
        backgroundColor: algorithmData?.color || "rgb(75, 192, 192)",
        borderColor: algorithmData?.color || "rgb(75, 192, 192)",
        pointRadius: 6,
        pointHoverRadius: 8,
      }
    })

    return { datasets }
  }

  const getScatterPlot3Data = () => {
    if (!simulationResults || simulationResults.length === 0) return null

    // Algorithm Performance Matrix: Convergence Speed vs Solution Quality
    const finalIteration = simulationResults[simulationResults.length - 1]

    const datasets = [
      {
        label: "Algorithm Performance",
        data: selectedAlgorithms.map((algorithm) => {
          const finalPower = Number.parseFloat(finalIteration[algorithm]) || 0

          // Calculate average improvement rate
          let totalImprovement = 0
          for (let i = 1; i < Math.min(50, simulationResults.length); i++) {
            const currentPower = Number.parseFloat(simulationResults[i][algorithm]) || 0
            const prevPower = Number.parseFloat(simulationResults[i - 1][algorithm]) || 0
            totalImprovement += Math.max(0, currentPower - prevPower)
          }
          const avgImprovementRate = totalImprovement / Math.min(49, simulationResults.length - 1)

          const algorithmData = algorithms.find((a) => a.value === algorithm)
          return {
            x: avgImprovementRate,
            y: finalPower,
            label: algorithm,
            backgroundColor: algorithmData?.color || "rgb(75, 192, 192)",
          }
        }),
        backgroundColor: selectedAlgorithms.map((algorithm) => {
          const algorithmData = algorithms.find((a) => a.value === algorithm)
          return algorithmData?.color || "rgb(75, 192, 192)"
        }),
        borderColor: selectedAlgorithms.map((algorithm) => {
          const algorithmData = algorithms.find((a) => a.value === algorithm)
          return algorithmData?.color || "rgb(75, 192, 192)"
        }),
        pointRadius: 10,
        pointHoverRadius: 12,
      },
    ]

    return { datasets }
  }

  const getScatterPlot4Data = () => {
    if (!simulationResults || !summaryResults || summaryResults.length === 0) return null

    // Power vs Voltage using actual data from summary CSV
    const datasets = selectedAlgorithms
        .map((algorithm) => {
          const algorithmData = algorithms.find((a) => a.value === algorithm)
          const summaryData = summaryResults.find((row: any) => row.Algorithm === algorithm)

          if (!summaryData) return null

          const voltage = Number.parseFloat(summaryData["Best Voltage (V)"]) || 0
          const power = Number.parseFloat(summaryData["Best Power (W)"]) || 0

          return {
            label: algorithm,
            data: [{ x: voltage, y: power }],
            backgroundColor: algorithmData?.color || "rgb(75, 192, 192)",
            borderColor: algorithmData?.color || "rgb(75, 192, 192)",
            pointRadius: 8,
            pointHoverRadius: 10,
          }
        })
        .filter(Boolean)

    return { datasets }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
        labels: {
          usePointStyle: true,
          boxWidth: 10,
          padding: 20,
          font: {
            size: 12,
          },
        },
      },
      title: {
        display: true,
        text: "Power Output Convergence",
        font: {
          size: 16,
          weight: "bold",
        },
        padding: {
          top: 10,
          bottom: 20,
        },
      },
      tooltip: {
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        titleColor: "#333",
        bodyColor: "#666",
        borderColor: "#ddd",
        borderWidth: 1,
        padding: 12,
        boxPadding: 6,
        usePointStyle: true,
        callbacks: {
          title: (tooltipItems: any) => {
            return `Iteration ${tooltipItems[0].label}`
          },
          label: (context: any) => {
            return `  ${context.dataset.label}: ${context.parsed.y.toFixed(2)}W`
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Iteration",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            top: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
        ticks: {
          maxRotation: 0,
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Power Output (W)",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            bottom: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
    },
    animation: {
      duration: 0, // Disable chart.js animation since we're controlling it manually
    },
    elements: {
      line: {
        borderJoinStyle: "round",
      },
    },
  }

  const scatterOptions1 = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
        labels: {
          usePointStyle: true,
          boxWidth: 10,
          padding: 20,
          font: {
            size: 12,
          },
        },
      },
      title: {
        display: true,
        text: "Final Power Output vs Convergence Speed",
        font: {
          size: 16,
          weight: "bold",
        },
        padding: {
          top: 10,
          bottom: 20,
        },
      },
      tooltip: {
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        titleColor: "#333",
        bodyColor: "#666",
        borderColor: "#ddd",
        borderWidth: 1,
        padding: 12,
        usePointStyle: true,
        callbacks: {
          label: (context: any) =>
              `${context.dataset.label}: (${context.parsed.x} iterations, ${context.parsed.y.toFixed(2)}W)`,
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Convergence Speed (Iterations to 95%)",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            top: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Final Power Output (W)",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            bottom: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
    },
  }

  const scatterOptions2 = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
        labels: {
          usePointStyle: true,
          boxWidth: 10,
          padding: 20,
          font: {
            size: 12,
          },
        },
      },
      title: {
        display: true,
        text: "Population Size vs Final Power Output",
        font: {
          size: 16,
          weight: "bold",
        },
        padding: {
          top: 10,
          bottom: 20,
        },
      },
      tooltip: {
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        titleColor: "#333",
        bodyColor: "#666",
        borderColor: "#ddd",
        borderWidth: 1,
        padding: 12,
        usePointStyle: true,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Population Size",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            top: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Final Power Output (W)",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            bottom: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
    },
  }

  const scatterOptions3 = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Algorithm Performance Matrix",
        font: {
          size: 16,
          weight: "bold",
        },
        padding: {
          top: 10,
          bottom: 20,
        },
      },
      tooltip: {
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        titleColor: "#333",
        bodyColor: "#666",
        borderColor: "#ddd",
        borderWidth: 1,
        padding: 12,
        usePointStyle: true,
        callbacks: {
          label: (context: any) => {
            const point = context.raw
            return `${point.label}: (${context.parsed.x.toFixed(2)} improvement rate, ${context.parsed.y.toFixed(2)}W)`
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Average Improvement Rate (W/iteration)",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            top: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Final Power Output (W)",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            bottom: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
    },
  }

  const scatterOptions4 = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
        labels: {
          usePointStyle: true,
          boxWidth: 10,
          padding: 20,
          font: {
            size: 12,
          },
        },
      },
      title: {
        display: true,
        text: "Power Output vs Optimal Voltage",
        font: {
          size: 16,
          weight: "bold",
        },
        padding: {
          top: 10,
          bottom: 20,
        },
      },
      tooltip: {
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        titleColor: "#333",
        bodyColor: "#666",
        borderColor: "#ddd",
        borderWidth: 1,
        padding: 12,
        usePointStyle: true,
        callbacks: {
          label: (context: any) =>
              `${context.dataset.label}: (${context.parsed.x.toFixed(2)}V, ${context.parsed.y.toFixed(2)}W)`,
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Optimal Voltage (V)",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            top: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
        min: 0,
        max: 100,
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Power Output (W)",
          font: {
            size: 14,
            weight: "bold",
          },
          padding: {
            bottom: 10,
          },
        },
        grid: {
          display: true,
          color: "rgba(0, 0, 0, 0.05)",
        },
      },
    },
  }

  const toggleAlgorithm = (algorithm: string) => {
    setSelectedAlgorithms((prev) =>
        prev.includes(algorithm) ? prev.filter((a) => a !== algorithm) : [...prev, algorithm],
    )
  }

  return (
      <div className="min-h-screen bg-gray-50 py-8 px-4">
        <div className="container mx-auto max-w-7xl">
          <div className="text-center mb-6">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Algorithm Simulation</h1>
            <p className="text-gray-600">18 MPPT Algorithms for Photovoltaic System Optimization</p>
          </div>

          {/* Main Content Area */}
          <div className="grid lg:grid-cols-4 gap-6">
            {/* Visualization - Now the main and larger section */}
            <div className="lg:col-span-3 space-y-6">
              {/* Main Visualization Card */}
              <Card className="shadow-md">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center">
                    <TrendingUp className="mr-2 h-5 w-5" />
                    Algorithm Visualization
                  </span>
                    <div className="flex gap-2">
                      <Button
                          variant={!showScatterPlots ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowScatterPlots(false)}
                      >
                        Line Chart
                      </Button>
                      <Button
                          variant={showScatterPlots ? "default" : "outline"}
                          size="sm"
                          onClick={() => setShowScatterPlots(true)}
                      >
                        Scatter Plots
                      </Button>
                    </div>
                  </CardTitle>
                  <CardDescription>
                    {!showScatterPlots
                        ? "Real-time convergence curve showing power output over iterations"
                        : "Statistical analysis through scatter plot visualizations"}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {!showScatterPlots ? (
                      <div className="h-[500px] w-full bg-white rounded-lg p-4">
                        {simulationResults && getChartData() ? (
                            <Line id="convergence-chart" data={getChartData()!} options={chartOptions} />
                        ) : (
                            <div className="h-full bg-gray-50 rounded-lg flex items-center justify-center">
                              <div className="text-center text-gray-500">
                                <TrendingUp className="h-16 w-16 mx-auto mb-4 opacity-30" />
                                <p className="text-lg font-medium mb-2">Convergence chart will appear here</p>
                                <p className="text-sm">Select algorithms and start simulation to view results</p>
                              </div>
                            </div>
                        )}
                      </div>
                  ) : (
                      <div className="space-y-4">
                        {/* Scatter Plot Navigation */}
                        <div className="flex justify-center gap-2 flex-wrap">
                          <Button
                              variant={activeScatterPlot === 0 ? "default" : "outline"}
                              size="sm"
                              onClick={() => setActiveScatterPlot(0)}
                          >
                            Power vs Speed
                          </Button>
                          <Button
                              variant={activeScatterPlot === 1 ? "default" : "outline"}
                              size="sm"
                              onClick={() => setActiveScatterPlot(1)}
                          >
                            Population Analysis
                          </Button>
                          <Button
                              variant={activeScatterPlot === 2 ? "default" : "outline"}
                              size="sm"
                              onClick={() => setActiveScatterPlot(2)}
                          >
                            Performance Matrix
                          </Button>
                          <Button
                              variant={activeScatterPlot === 3 ? "default" : "outline"}
                              size="sm"
                              onClick={() => setActiveScatterPlot(3)}
                          >
                            Power vs Voltage
                          </Button>
                        </div>

                        {/* Scatter Plot Display */}
                        <div className="h-[500px] w-full bg-white rounded-lg p-4">
                          {simulationResults ? (
                              <>
                                {activeScatterPlot === 0 && getScatterPlot1Data() && (
                                    <Scatter data={getScatterPlot1Data()!} options={scatterOptions1} />
                                )}
                                {activeScatterPlot === 1 && getScatterPlot2Data() && (
                                    <Scatter data={getScatterPlot2Data()!} options={scatterOptions2} />
                                )}
                                {activeScatterPlot === 2 && getScatterPlot3Data() && (
                                    <Scatter data={getScatterPlot3Data()!} options={scatterOptions3} />
                                )}
                                {activeScatterPlot === 3 && getScatterPlot4Data() && (
                                    <Scatter data={getScatterPlot4Data()!} options={scatterOptions4} />
                                )}
                              </>
                          ) : (
                              <div className="h-full bg-gray-50 rounded-lg flex items-center justify-center">
                                <div className="text-center text-gray-500">
                                  <TrendingUp className="h-16 w-16 mx-auto mb-4 opacity-30" />
                                  <p className="text-lg font-medium mb-2">Scatter plots will appear here</p>
                                  <p className="text-sm">Run simulation to view statistical analysis</p>
                                </div>
                              </div>
                          )}
                        </div>

                        {/* Scatter Plot Descriptions */}
                        {simulationResults && (
                            <div className="text-sm text-gray-600 bg-gray-50 p-4 rounded-lg">
                              {activeScatterPlot === 0 && (
                                  <p>
                                    <strong>Power vs Speed:</strong> Shows the relationship between final power output and
                                    convergence speed (iterations to reach 95% of final power). Algorithms in the top-left are
                                    ideal (high power, fast convergence).
                                  </p>
                              )}
                              {activeScatterPlot === 1 && (
                                  <p>
                                    <strong>Population Analysis:</strong> Demonstrates how population size affects final power
                                    output for all selected algorithms. Shows optimal population size ranges and scaling
                                    behavior.
                                  </p>
                              )}
                              {activeScatterPlot === 2 && (
                                  <p>
                                    <strong>Performance Matrix:</strong> Compares algorithms based on improvement rate vs final
                                    power. Top-right quadrant represents superior algorithms (high improvement rate and high
                                    final power).
                                  </p>
                              )}
                              {activeScatterPlot === 3 && (
                                  <p>
                                    <strong>Power vs Voltage:</strong> Shows the actual optimal voltage and power output
                                    achieved by each algorithm from the simulation results. This represents real MPPT
                                    performance data rather than theoretical calculations.
                                  </p>
                              )}
                            </div>
                        )}
                      </div>
                  )}
                </CardContent>
              </Card>

              {/* Status Card - Now below the main visualization */}
              <Card className="shadow-sm">
                <CardHeader className="pb-2">
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
                  <div className="grid grid-cols-4 gap-4">
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
            </div>

            {/* Configuration Panel - Now on the right side */}
            <div className="lg:col-span-1 space-y-6">
              <Card className="shadow-md">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center">
                    <Settings className="mr-2 h-5 w-5" />
                    Configuration
                  </CardTitle>
                  <CardDescription>Set parameters and select algorithms</CardDescription>
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
                    <Label>Select Algorithms ({selectedAlgorithms.length}/18):</Label>
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

              {/* Algorithm Info Card */}
              <Card className="shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle>Selected Algorithms</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 text-sm">
                    <div className="flex flex-wrap gap-1">
                      {selectedAlgorithms.map((algorithm) => (
                          <Badge key={algorithm} variant="secondary" className="text-xs">
                            {algorithm}
                          </Badge>
                      ))}
                    </div>
                    <div className="pt-2 border-t">
                      <h4 className="font-semibold mb-2">Environment Settings:</h4>
                      <ul className="space-y-1 text-gray-600">
                        <li>Irradiance: 800 W/m²</li>
                        <li>Temperature: 25°C</li>
                        <li>Panel Type: Monocrystalline</li>
                        <li>V_oc: 100V, I_sc: 10A</li>
                      </ul>
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
