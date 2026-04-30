<script setup lang="ts">
import { reactive, ref, nextTick, watch } from 'vue'
// import * as Plotly from 'plotly.js-dist'
import * as Plotly from 'plotly.js'
import './App.css'
import LatexEquation from './components/LatexEquation.vue'

// Constants
const API_URL = import.meta.env.VITE_API_BASE_URL
console.log(API_URL)
const DEBOUNCE_MS = 100
const CSV_THRESHOLD = 0.01
const SLIDER_STEP = 0.01

// Colors for plots
const COLORS = {
  normalized: '#2196F3',
  predicted: '#FF9800',
  tuned: '#4CAF50',
  input: '#FF00FF',
}

interface CSVRow {
  time: string
  stepResponse: string
  inputFunc: string | null
}

interface PredictResponse {
  pred_tftype: string
  tf_params: {
    K: number
    T1: number | null
    T2: number | null
    b: number | null
  }
  step_mag: number
  pred_step_response: number[]
  norm_step_response: number[]
  norm_t: number[]
  norm_in: number[] | null
}

const csvData = ref<CSVRow[]>([])
const fileName = ref<string>('')
const error = ref<string>('')
const loading = ref(false)
const prediction = ref<PredictResponse | null>(null)
const selectedTftype = ref<string>('auto')
const stepMagnitudeInput = ref('0')
const oscInput = ref(false)

const currStepMagnitude = ref<number>(1)

// Parameter sliders
const paramK = ref(1)
const paramT1 = ref(1)
const paramT2 = ref(1)
const paramB = ref(0.5)
const tuningPlotResponse = ref<number[]>([])
const Tmax = ref(1)
const lastUpdateTime = ref(0)

type ParamKey = 'K' | 'T1' | 'T2' | 'b'

const sliderRanges = reactive<Record<ParamKey, { min: number; max: number }>>({
  K: { min: 0.1, max: 10 },
  T1: { min: 0.1, max: 10 },
  T2: { min: 0.1, max: 10 },
  b: { min: 0.05, max: 1 },
})

const tfTypes = [
  { value: 'auto', label: 'Auto Identify' },
  { value: 'ORD1', label: '1st Order' },
  { value: 'ORD1_ASTAT', label: '1st Order (Astatic)' },
  { value: 'ORD2_APER', label: '2nd Order (Aperiodic)' },
  { value: 'ORD2_PER', label: '2nd Order (Periodic)' },
  { value: 'ORD2_ASTAT', label: '2nd Order (Astatic)' },
  { value: 'ORD2_ASTAT_T', label: '2nd Order (Astatic with T)' },
]

function getTfTypeLabel(tfTypeValue: string): string {
  const tfType = tfTypes.find((tf) => tf.value === tfTypeValue)
  return tfType ? tfType.label : tfTypeValue
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function validateStepMagnitudeInput(event: Event): void {
  const input = event.target as HTMLInputElement
  // Only allow digits and decimal point
  input.value = input.value.replace(/[^0-9.]/g, '')
  stepMagnitudeInput.value = input.value
}

function parseCSV(content: string): CSVRow[] {
  const lines = content
    .trim()
    .split('\n')
    .filter((line) => line.trim())
  if (lines.length === 0) return []

  // Parse first line to determine number of columns
  const firstValues = lines[0]!.split(',').map((v) => v.trim())
  const numColumns = firstValues.length

  // Enforce exactly number of columns
  if (numColumns !== 2 && numColumns !== 3) {
    throw new Error(`Expected 2 or 3 columns but found ${numColumns}`)
  }

  // Generate headers: Time and Response
  const headers = numColumns === 2 ? ['Time', 'Response'] : ['Time', 'Input', 'Response']

  // Parse all rows as data
  const rows: CSVRow[] = lines.map((line) => {
    const values = line.split(',').map((v) => v.trim())
    const row: CSVRow = {
      time: '',
      stepResponse: '',
      inputFunc: '',
    }
    headers.forEach(() => {
      row.time = values[0] || ''
      row.stepResponse = headers.includes('Input') ? values[2] || '' : values[1] || ''
      row.inputFunc = headers.includes('Input') ? values[1] || null : null
    })
    return row
  })

  return rows
}

function handleFileUpload(event: Event): void {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]

  if (!file) return

  // Validate file type
  if (!file.name.endsWith('.csv')) {
    error.value = 'Please upload a CSV file'
    csvData.value = []
    return
  }

  fileName.value = file.name
  error.value = ''
  prediction.value = null

  // Read file
  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const content = e.target?.result as string
      csvData.value = parseCSV(content)
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Error parsing CSV file'
      csvData.value = []
    }
  }
  reader.onerror = () => {
    error.value = 'Error reading file'
    csvData.value = []
  }
  reader.readAsText(file)
}

let usedOscInpForPred = false

async function handlePredict(): Promise<void> {
  if (csvData.value.length === 0) {
    error.value = 'No data loaded'
    return
  }

  loading.value = true
  error.value = ''

  try {
    // Extract step response values and max time
    const stepResponse = csvData.value.map((row) => parseFloat(row.stepResponse || '0'))
    const timeValues = csvData.value.map((row) => parseFloat(row.time || '0'))

    const inputFunc =
      csvData.value && csvData.value[0]?.inputFunc
        ? csvData.value.map((row) => parseFloat(row.inputFunc || '0'))
        : null
    // if (csvData.value && csvData.value[0]?.inputFunc) {
    //   const inputFunc = csvData.value.map((row) => parseFloat(row.inputFunc || '0'))
    // }

    Tmax.value = Math.max(...timeValues)

    usedOscInpForPred = oscInput.value

    // Call predict endpoint
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        step_response: stepResponse,
        Tmax: Tmax.value,
        tftype: selectedTftype.value === 'auto' ? null : selectedTftype.value,
        strip_zeros: true,
        strip_threshold: CSV_THRESHOLD,
        step_mag: parseFloat(stepMagnitudeInput.value),
        input_func: inputFunc,
        osc_inp: usedOscInpForPred,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.error || `Server error: ${response.status}`)
    }

    const data: PredictResponse = await response.json()
    prediction.value = data
    currStepMagnitude.value = data.step_mag

    // Initialize parameter sliders with predicted values
    paramK.value = data.tf_params.K
    if (data.tf_params.T1 !== null) paramT1.value = data.tf_params.T1
    if (data.tf_params.T2 !== null) paramT2.value = data.tf_params.T2
    if (data.tf_params.b !== null) paramB.value = data.tf_params.b

    // Initialize slider ranges based on predicted values
    sliderRanges.K.min = Math.min(0.1, data.tf_params.K)
    sliderRanges.K.max = Math.max(10, data.tf_params.K)

    if (data.tf_params.T1 !== null) {
      sliderRanges.T1.min = Math.min(0.1, data.tf_params.T1)
      sliderRanges.T1.max = Math.max(10, data.tf_params.T1)
    }
    if (data.tf_params.T2 !== null) {
      sliderRanges.T2.min = Math.min(0.1, data.tf_params.T2)
      sliderRanges.T2.max = Math.max(10, data.tf_params.T2)
    }
    if (data.tf_params.b !== null) {
      sliderRanges.b.min = Math.min(0.05, data.tf_params.b)
      sliderRanges.b.max = Math.max(1, data.tf_params.b)
    }

    // Wait for DOM to update before plotting
    await nextTick()
    plotResults(data, false)
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Error making prediction'
    prediction.value = null
  } finally {
    loading.value = false
  }
}

function plotResults(data: PredictResponse, isTuned: boolean = false): void {
  const normalizedTrace: Plotly.Data = {
    x: data.norm_t,
    y: data.norm_step_response,
    type: 'scatter',
    mode: 'lines',
    name: 'Observed step response',
    line: { color: COLORS.normalized, width: 2 },
  }

  const predictedTrace: Plotly.Data = {
    x: data.norm_t,
    y: data.pred_step_response,
    type: 'scatter',
    mode: 'lines',
    name: 'Predicted step response',
    line: { color: COLORS.predicted, width: 2, dash: 'dash' },
  }

  const traces: Array<Plotly.Data> = [normalizedTrace, predictedTrace]

  if (data.norm_in && data.norm_in.length > 0) {
    const inputTrace: Plotly.Data = {
      x: data.norm_t,
      y: data.norm_in,
      type: 'scatter',
      mode: 'lines',
      name: 'Input signal',
      line: { color: COLORS.input, width: 2 },
    }
    traces.push(inputTrace)
  }

  // Add tuned trace if available
  if (isTuned && tuningPlotResponse.value.length > 0) {
    const tunedTrace: Plotly.Data = {
      x: data.norm_t,
      y: tuningPlotResponse.value,
      type: 'scatter',
      mode: 'lines',
      name: 'Tuned step response',
      line: { color: COLORS.tuned, width: 2 },
    }
    traces.push(tunedTrace)
  }

  const layout: Partial<Plotly.Layout> = {
    title: {
      // text: `System Identification - ${isTuned ? 'Tuning' : 'Predicted TF Type'}: ${data.pred_tftype}`,
    },
    xaxis: { title: { text: 'Time [s]' } },
    yaxis: { title: { text: 'Level' } },
    hovermode: 'closest',
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#fff',
    font: { family: 'Arial, sans-serif', size: 12 },
    legend: {
      x: 0.5,
      y: 1.1,
      xanchor: 'center',
      yanchor: 'bottom',
      orientation: 'h',
    },
  }

  Plotly.newPlot('plot', traces, layout, { responsive: true })
}

async function updateTunedStepResponse(): Promise<void> {
  if (!prediction.value) return

  // Debounce: ignore if called within DEBOUNCE_MS of last call
  const now = Date.now()
  if (now - lastUpdateTime.value < DEBOUNCE_MS) return
  lastUpdateTime.value = now

  try {
    const response = await fetch(`${API_URL}/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tftype: prediction.value.pred_tftype,
        params: {
          K: paramK.value * currStepMagnitude.value,
          T1: prediction.value.tf_params.T1 !== null ? paramT1.value : null,
          T2: prediction.value.tf_params.T2 !== null ? paramT2.value : null,
          b: prediction.value.tf_params.b !== null ? paramB.value : null,
        },
        Tmax: Tmax.value,
        Tmin: prediction.value.norm_t[0],
        step_mag: currStepMagnitude.value,
        input_func: prediction.value.norm_in,
      }),
    })

    if (!response.ok) throw new Error('Error calling /step endpoint')

    const data = await response.json()
    tuningPlotResponse.value = data.step_response

    // Update plot with tuned response
    await nextTick()
    plotResults(prediction.value, true)
  } catch (err) {
    console.error('Error updating tuned response:', err)
  }
}

watch(
  () => [sliderRanges.K.min, sliderRanges.K.max],
  () => {
    if (sliderRanges.K.max < sliderRanges.K.min) sliderRanges.K.max = sliderRanges.K.min
    const clamped = clamp(paramK.value, sliderRanges.K.min, sliderRanges.K.max)
    if (clamped !== paramK.value) {
      paramK.value = clamped
      updateTunedStepResponse()
    }
  },
)

watch(
  () => [sliderRanges.T1.min, sliderRanges.T1.max],
  () => {
    if (prediction.value?.tf_params.T1 === null) return
    if (sliderRanges.T1.max < sliderRanges.T1.min) sliderRanges.T1.max = sliderRanges.T1.min
    const clamped = clamp(paramT1.value, sliderRanges.T1.min, sliderRanges.T1.max)
    if (clamped !== paramT1.value) {
      paramT1.value = clamped
      updateTunedStepResponse()
    }
  },
)

watch(
  () => [sliderRanges.T2.min, sliderRanges.T2.max],
  () => {
    if (prediction.value?.tf_params.T2 === null) return
    if (sliderRanges.T2.max < sliderRanges.T2.min) sliderRanges.T2.max = sliderRanges.T2.min
    const clamped = clamp(paramT2.value, sliderRanges.T2.min, sliderRanges.T2.max)
    if (clamped !== paramT2.value) {
      paramT2.value = clamped
      updateTunedStepResponse()
    }
  },
)

watch(
  () => [sliderRanges.b.min, sliderRanges.b.max],
  () => {
    if (prediction.value?.tf_params.b === null) return
    if (sliderRanges.b.max < sliderRanges.b.min) sliderRanges.b.max = sliderRanges.b.min
    const clamped = clamp(paramB.value, sliderRanges.b.min, sliderRanges.b.max)
    if (clamped !== paramB.value) {
      paramB.value = clamped
      updateTunedStepResponse()
    }
  },
)
</script>

<template>
  <main class="px-10 pt-4 w-lvw">
    <h1 class="text-3xl text-black font-bold mb-4">Control System Identification</h1>
    <div class="flex flex-row">
      <!-- Left control panel -->
      <div class="flex flex-col gap-4 w-1/3">
        <!-- Upload section -->
        <div class="flex flex-col min-w-full">
          <input
            type="file"
            accept=".csv"
            @change="handleFileUpload"
            class="min-w-full border-2 p-1 cursor-pointer text-md font-bold text-blue-600 hover:bg-blue-600 hover:text-white"
          />
          <div v-if="error" class="italic text-md text-red-400">{{ error }}</div>
          <div v-if="csvData.length > 0" class="italic text-md text-blue-500">
            ✓ Data loaded successfully ({{ csvData.length }} rows)
          </div>
        </div>

        <!-- Type selection section -->
        <div v-if="csvData.length > 0" class="pe-0.5 gap-4 flex flex-col">
          <div>
            <label>Input signal magnitude (0 for auto):</label>
            <input
              type="text"
              v-model="stepMagnitudeInput"
              @input="validateStepMagnitudeInput"
              inputmode="decimal"
              class="min-w-full"
            />
          </div>

          <div>
            <label for="osc-toggle" class="cursor-pointer pe-4">Square signal input:</label>
            <input
              id="osc-toggle"
              type="checkbox"
              v-model="oscInput"
              class="w-4 h-4 border border-default-medium rounded-xs bg-neutral-secondary-medium focus:ring-2 focus:ring-blue-300 cursor-pointer"
            />
          </div>

          <div>
            <label>System order:</label>
            <select v-model="selectedTftype" id="tftype-select" class="w-full">
              <option v-for="tf in tfTypes" :key="tf.value" :value="tf.value">
                {{ tf.label }}
              </option>
            </select>
          </div>
          <button
            @click="handlePredict"
            :disabled="loading"
            class="min-w-full p-1 border-2 text-blue-600 text-md font-bold cursor-pointer hover:bg-blue-600 hover:text-white"
          >
            Identify
          </button>
        </div>

        <!-- Results section -->
        <div v-if="prediction" class="flex flex-col">
          <!-- ID'd TF type section -->
          <label>Identified system:</label>
          <div class="flex flex-col gap-4 ps-4">
            <p class="font-bold text-xl text-blue-600 wrap-normal">
              {{ getTfTypeLabel(prediction.pred_tftype) }}
            </p>
            <LatexEquation
              v-if="prediction.pred_tftype == 'ORD1'"
              equation="F(s)=\frac{K}{T_{1}s+1}"
            />
            <LatexEquation
              v-if="prediction.pred_tftype == 'ORD1_ASTAT'"
              equation="F(s)=\frac{K}{s}"
            />
            <LatexEquation
              v-if="prediction.pred_tftype == 'ORD2_APER'"
              equation="F(s)=\frac{K}{(T_{1}s+1)(T_{2}s+1)}"
            />
            <LatexEquation
              v-if="prediction.pred_tftype == 'ORD2_PER'"
              equation="F(s)=\frac{K}{T_{1}^{2}s^{2}+2b T_{1}s+1}"
            />
            <LatexEquation
              v-if="prediction.pred_tftype == 'ORD2_ASTAT'"
              equation="F(s)=\frac{K}{s^2}"
            />
            <LatexEquation
              v-if="prediction.pred_tftype == 'ORD2_ASTAT_T'"
              equation="F(s)=\frac{K}{s(T_{1}s+1)}"
            />
            <!-- ID'd parameters section -->
            <div class="grid grid-cols-[1fr_5fr]">
              <p>K:</p>
              <div>{{ prediction.tf_params.K.toFixed(4) }}</div>
              <p v-if="prediction.tf_params.T1 !== null">T1:</p>
              <div v-if="prediction.tf_params.T1 !== null">
                {{ prediction.tf_params.T1.toFixed(4) }}
              </div>
              <p v-if="prediction.tf_params.T2 !== null">T2:</p>
              <div v-if="prediction.tf_params.T2 !== null">
                {{ prediction.tf_params.T2.toFixed(4) }}
              </div>
              <p v-if="prediction.tf_params.b !== null">b:</p>
              <div v-if="prediction.tf_params.b !== null">
                {{ prediction.tf_params.b.toFixed(4) }}
              </div>
            </div>
          </div>
        </div>
      </div>
      <div id="plot" class="w-full"></div>
    </div>

    <!-- Fine-tuning sliders section -->
    <div v-if="prediction" class="pt-4">
      <label>Fine-tune parameters:</label>
      <div class="mt-2 ps-4 flex flex-row gap-4 flex-wrap">
        <!-- K Slider -->
        <div class="flex-1 min-w-[250px]">
          <div class="flex flex-row text-xl">
            <span class="me-4">K:</span>
            <span>{{ paramK.toFixed(4) }}</span>
          </div>
          <div class="grid grid-cols-[auto_auto_auto_1fr_auto_auto_auto] gap-1 items-center">
            <span>Min:</span>
            <input
              type="number"
              v-model.number="sliderRanges.K.min"
              step="0.1"
              aria-label="Minimum K"
            />
            <button
              @click="
                () => {
                  paramK = clamp(paramK - SLIDER_STEP, sliderRanges.K.min, sliderRanges.K.max)
                  updateTunedStepResponse()
                }
              "
              class="arrow-btn"
            >
              {{ '<' }}
            </button>
            <input
              type="range"
              v-model.number="paramK"
              @input="updateTunedStepResponse"
              :min="sliderRanges.K.min"
              :max="sliderRanges.K.max"
              :step="SLIDER_STEP"
              class="accent-blue-600"
            />
            <button
              @click="
                () => {
                  paramK = clamp(paramK + SLIDER_STEP, sliderRanges.K.min, sliderRanges.K.max)
                  updateTunedStepResponse()
                }
              "
              class="arrow-btn"
            >
              {{ '>' }}
            </button>
            <span>Max:</span>
            <input
              type="number"
              v-model.number="sliderRanges.K.max"
              step="0.1"
              aria-label="Maximum K"
            />
          </div>
        </div>

        <!-- T1 Slider -->
        <div v-if="prediction.tf_params.T1 !== null" class="flex-1 min-w-[250px] contbox">
          <div class="flex flex-row text-xl">
            <span class="me-4">T1:</span>
            <span>{{ paramT1.toFixed(4) }}</span>
          </div>
          <div class="grid grid-cols-[auto_auto_auto_1fr_auto_auto_auto] gap-1 items-center">
            <span>Min:</span>
            <input
              type="number"
              v-model.number="sliderRanges.T1.min"
              step="0.1"
              aria-label="Minimum T1"
            />
            <button
              @click="
                () => {
                  paramT1 = clamp(paramT1 - SLIDER_STEP, sliderRanges.T1.min, sliderRanges.T1.max)
                  updateTunedStepResponse()
                }
              "
              class="arrow-btn"
            >
              {{ '<' }}
            </button>
            <input
              type="range"
              v-model.number="paramT1"
              @input="updateTunedStepResponse"
              :min="sliderRanges.T1.min"
              :max="sliderRanges.T1.max"
              :step="SLIDER_STEP"
              class="accent-blue-600"
            />
            <button
              @click="
                () => {
                  paramT1 = clamp(paramT1 + SLIDER_STEP, sliderRanges.T1.min, sliderRanges.T1.max)
                  updateTunedStepResponse()
                }
              "
              class="arrow-btn"
            >
              {{ '>' }}
            </button>
            <span>Max:</span>
            <input
              type="number"
              v-model.number="sliderRanges.T1.max"
              step="0.1"
              aria-label="Maximum T1"
            />
          </div>
        </div>

        <!-- T2 Slider -->
        <div v-if="prediction.tf_params.T2 !== null" class="flex-1 min-w-[250px] contbox">
          <div class="flex flex-row text-xl">
            <span class="me-4">T2:</span>
            <span>{{ paramT2.toFixed(4) }}</span>
          </div>
          <div class="grid grid-cols-[auto_auto_auto_1fr_auto_auto_auto] gap-1 items-center">
            <span>Min:</span>
            <input
              type="number"
              v-model.number="sliderRanges.T2.min"
              step="0.1"
              aria-label="Minimum T2"
            />
            <button
              @click="
                () => {
                  paramT2 = clamp(paramT2 - SLIDER_STEP, sliderRanges.T2.min, sliderRanges.T2.max)
                  updateTunedStepResponse()
                }
              "
              class="arrow-btn"
            >
              {{ '<' }}
            </button>
            <input
              type="range"
              v-model.number="paramT2"
              @input="updateTunedStepResponse"
              :min="sliderRanges.T2.min"
              :max="sliderRanges.T2.max"
              :step="SLIDER_STEP"
              class="accent-blue-600"
            />
            <button
              @click="
                () => {
                  paramT2 = clamp(paramT2 + SLIDER_STEP, sliderRanges.T2.min, sliderRanges.T2.max)
                  updateTunedStepResponse()
                }
              "
              class="arrow-btn"
            >
              {{ '>' }}
            </button>
            <span>Max:</span>
            <input
              type="number"
              v-model.number="sliderRanges.T2.max"
              step="0.1"
              aria-label="Maximum T2"
            />
          </div>
        </div>

        <!-- b Slider -->
        <div v-if="prediction.tf_params.b !== null" class="flex-1 min-w-[250px] contbox">
          <div class="flex flex-row text-xl">
            <span class="me-4">b:</span>
            <span>{{ paramB.toFixed(4) }}</span>
          </div>
          <div class="grid grid-cols-[auto_auto_auto_1fr_auto_auto_auto] gap-1 items-center">
            <span>Min:</span>
            <input
              type="number"
              v-model.number="sliderRanges.b.min"
              step="0.1"
              aria-label="Minimum b"
            />
            <button
              @click="
                () => {
                  paramB = clamp(paramB - SLIDER_STEP, sliderRanges.b.min, sliderRanges.b.max)
                  updateTunedStepResponse()
                }
              "
              class="arrow-btn"
            >
              {{ '<' }}
            </button>
            <input
              type="range"
              v-model.number="paramB"
              @input="updateTunedStepResponse"
              :min="sliderRanges.b.min"
              :max="sliderRanges.b.max"
              :step="SLIDER_STEP"
              class="accent-blue-600"
            />
            <button
              @click="
                () => {
                  paramB = clamp(paramB + SLIDER_STEP, sliderRanges.b.min, sliderRanges.b.max)
                  updateTunedStepResponse()
                }
              "
              class="arrow-btn"
            >
              {{ '>' }}
            </button>
            <span>Max:</span>
            <input
              type="number"
              v-model.number="sliderRanges.b.max"
              step="0.1"
              aria-label="Maximum b"
            />
          </div>
        </div>
      </div>
    </div>
  </main>
</template>
