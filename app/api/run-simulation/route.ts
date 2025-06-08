import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import path from "path"
import fs from "fs"
import csv from "csv-parser"

export async function POST(request: NextRequest) {
  try {
    const { pop_size, max_iter } = await request.json()

    // Validate inputs
    if (!pop_size || !max_iter || pop_size < 10 || max_iter < 50) {
      return NextResponse.json(
          { error: "Invalid parameters. Population size must be >= 10 and max iterations >= 50" },
          { status: 400 },
      )
    }

    // Create a unique directory for this simulation
    const simulationId = Date.now().toString()
    const outputDir = path.join(process.cwd(), "temp", simulationId)

    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true })
    }

    // Copy the Python script to the output directory
    const scriptPath = path.join(process.cwd(), "scripts", "Full_algorithm_extractor_tables_cli_v2.py")
    const outputScriptPath = path.join(outputDir, "Full_algorithm_extractor_tables_cli_v2.py")

    if (!fs.existsSync(scriptPath)) {
      return NextResponse.json({ error: "Python script not found" }, { status: 500 })
    }

    fs.copyFileSync(scriptPath, outputScriptPath)

    // Run the Python script
    const pythonProcess = spawn(
        "python3",
        [outputScriptPath, "--pop_size", pop_size.toString(), "--max_iter", max_iter.toString()],
        {
          cwd: outputDir,
          stdio: ["pipe", "pipe", "pipe"],
        },
    )

    let stdout = ""
    let stderr = ""

    pythonProcess.stdout.on("data", (data) => {
      stdout += data.toString()
    })

    pythonProcess.stderr.on("data", (data) => {
      stderr += data.toString()
    })

    // Wait for the process to complete
    await new Promise((resolve, reject) => {
      pythonProcess.on("close", (code) => {
        if (code === 0) {
          resolve(code)
        } else {
          reject(new Error(`Python script exited with code ${code}: ${stderr}`))
        }
      })
    })

    // Read the power_vs_iteration.csv file
    const csvPath = path.join(outputDir, "power_vs_iteration.csv")
    const summaryPath = path.join(outputDir, "summary_metrics.csv")

    if (!fs.existsSync(csvPath)) {
      return NextResponse.json({ error: "Results file not found" }, { status: 500 })
    }

    const results = await readCSV(csvPath)
    const summary = fs.existsSync(summaryPath) ? await readCSV(summaryPath) : []

    // Clean up temporary files
    setTimeout(() => {
      fs.rmSync(outputDir, { recursive: true, force: true })
    }, 60000) // Clean up after 1 minute

    return NextResponse.json({
      success: true,
      simulationId,
      results,
      summary,
      stdout,
    })
  } catch (error) {
    console.error("Simulation error:", error)
    return NextResponse.json({ error: "Failed to run simulation" }, { status: 500 })
  }
}

async function readCSV(filePath: string): Promise<any[]> {
  return new Promise((resolve, reject) => {
    const results: any[] = []
    fs.createReadStream(filePath)
        .pipe(csv())
        .on("data", (data) => results.push(data))
        .on("end", () => resolve(results))
        .on("error", reject)
  })
}
