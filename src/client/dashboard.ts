import { Chart, LineController, LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend, Filler } from 'chart.js';
import type { TrainerMetrics } from '../distributed/trainer';

Chart.register(LineController, LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend, Filler);

// Styling theme
const colors = {
  bg: '#141620',
  grid: 'rgba(255, 255, 255, 0.05)',
  text: '#8b8fa4',
  loss: '#34d399',
  smoothLoss: '#22d3ee',
  gradNorm: '#a78bfa',
  peers: '#fb923c',
  tps: '#4f8cff'
};

export class Dashboard {
  metricsChart: Chart | null = null;
  history: {
    steps: number[],
    loss: number[],
    smoothLoss: number[],
    gradNorm: number[],
    peers: number[],
    tps: number[],
    arTime: number[]
  } = {
    steps: [], loss: [], smoothLoss: [], gradNorm: [], peers: [], tps: [], arTime: []
  };

  constructor(canvasId: string) {
    this.initChart(canvasId);
  }

  private initChart(canvasId: string) {
    const ctx = (document.getElementById(canvasId) as HTMLCanvasElement)?.getContext('2d');
    if (!ctx) return;

    this.metricsChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [], // Steps
        datasets: [
          {
            label: 'Smoothed Loss',
            data: [],
            borderColor: colors.smoothLoss,
            backgroundColor: colors.smoothLoss + '20',
            borderWidth: 2,
            pointRadius: 0,
            pointHitRadius: 5,
            fill: true,
            yAxisID: 'yLoss',
            tension: 0.4
          },
          {
            label: 'Raw Loss',
            data: [],
            borderColor: colors.loss + '40',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            yAxisID: 'yLoss',
            tension: 0.1
          },
          {
            label: 'Gradient Norm',
            data: [],
            borderColor: colors.gradNorm,
            borderWidth: 1.5,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
            yAxisID: 'yNorm',
            tension: 0.4
          },
          {
            label: 'Peers Context',
            data: [],
            borderColor: colors.peers,
            borderWidth: 1,
            pointRadius: 0,
            fill: true,
            backgroundColor: colors.peers + '10',
            yAxisID: 'yPeers',
            stepped: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            labels: { color: colors.text, boxWidth: 12, usePointStyle: true }
          },
          tooltip: {
            backgroundColor: 'rgba(10, 11, 15, 0.9)',
            titleColor: '#fff',
            bodyColor: '#e8eaf0',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            padding: 10
          }
        },
        scales: {
          x: {
            ticks: { color: colors.text, maxRotation: 0, maxTicksLimit: 8 },
            grid: { color: colors.grid }
          },
          yLoss: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Cross-Entropy Loss', color: colors.text },
            ticks: { color: colors.text },
            grid: { color: colors.grid }
          },
          yNorm: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'L2 Grad Norm', color: colors.gradNorm },
            ticks: { color: colors.gradNorm },
            grid: { display: false },
            min: 0
          },
          yPeers: {
            type: 'linear',
            position: 'right',
            title: { display: false },
            ticks: { display: false },
            grid: { display: false },
            min: 0,
            max: 20 // scale to bottom of chart
          }
        }
      }
    });
  }

  update(m: TrainerMetrics) {
    if (m.loss <= 0 || m.step === 0) return;

    this.history.steps.push(m.step);
    this.history.loss.push(m.loss);
    this.history.smoothLoss.push(m.smoothLoss);
    this.history.gradNorm.push(m.gradNorm);
    this.history.peers.push(m.peersConnected);
    this.history.tps.push(m.tokensPerSec);
    this.history.arTime.push(m.allReduceTimeMs);

    // Keep memory bounded to last 300 data points
    if (this.history.steps.length > 300) {
      this.history.steps.shift();
      this.history.loss.shift();
      this.history.smoothLoss.shift();
      this.history.gradNorm.shift();
      this.history.peers.shift();
      this.history.tps.shift();
      this.history.arTime.shift();
    }

    // Fast DOM updates
    this.setText('m-step', m.step.toString());
    this.setText('m-loss', m.loss.toFixed(6));
    this.setText('m-smooth-loss', `smooth: ${m.smoothLoss.toFixed(6)}`);
    this.setText('m-peers', m.peersConnected.toString());
    this.setText('m-grad-norm', m.gradNorm.toFixed(3));
    this.setText('m-tps', Math.floor(m.tokensPerSec).toLocaleString());
    this.setText('m-ar', m.allReduceTimeMs > 0 ? `${m.allReduceTimeMs.toFixed(0)} ms` : '— ms');

    // Chart.js update
    if (this.metricsChart) {
      this.metricsChart.data.labels = this.history.steps;
      this.metricsChart.data.datasets[0].data = this.history.smoothLoss;
      this.metricsChart.data.datasets[1].data = this.history.loss;
      this.metricsChart.data.datasets[2].data = this.history.gradNorm;
      this.metricsChart.data.datasets[3].data = this.history.peers;
      
      // Dynamic bounds mapping to focus on convergence rather than early noise
      const lossVals = this.history.smoothLoss.slice(Math.max(0, this.history.smoothLoss.length - 100));
      if (lossVals.length > 0) {
          const mMax = Math.max(...lossVals);
          const mMin = Math.min(...lossVals);
          // @ts-ignore
          this.metricsChart.options.scales.yLoss.max = mMax + (mMax - mMin) * 0.5;
          // @ts-ignore
          this.metricsChart.options.scales.yLoss.min = Math.max(0, mMin - (mMax - mMin) * 0.5);
      }

      this.metricsChart.update('none'); // Update without animation blocking
    }
  }

  private setText(id: string, text: string) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }
}
