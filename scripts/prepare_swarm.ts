/*
 * scripts/prepare_swarm.ts — Prepare data for swarm mode.
 *
 * Downloads OHLCV from Yahoo Finance, splits train/val, saves to public/data/
 * so the browser swarm worker can fetch it.
 *
 * Usage: pnpm run prepare:swarm
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT, 'public', 'data');

const TICKERS = ['SPY', 'QQQ', 'GLD', 'TLT', 'IWM'];
const START = '2010-01-01';
const TRAIN_END = '2020-12-31';
const VAL_END = '2022-12-31';

async function fetchTicker(ticker: string): Promise<{ date: string; close: number }[]> {
  const startEpoch = Math.floor(new Date(START).getTime() / 1000);
  const endEpoch = Math.floor(Date.now() / 1000);
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?period1=${startEpoch}&period2=${endEpoch}&interval=1d`;

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
      if (!res.ok) {
        if (res.status === 429) { await new Promise(r => setTimeout(r, 3000)); continue; }
        throw new Error(`HTTP ${res.status}`);
      }
      const json = await res.json() as any;
      const result = json.chart?.result?.[0];
      if (!result) throw new Error('No data');

      const timestamps: number[] = result.timestamp || [];
      const closes: (number | null)[] = result.indicators?.quote?.[0]?.close || [];
      const bars: { date: string; close: number }[] = [];
      for (let i = 0; i < timestamps.length; i++) {
        if (closes[i] != null && !isNaN(closes[i])) {
          bars.push({ date: new Date(timestamps[i] * 1000).toISOString().split('T')[0], close: closes[i] as number });
        }
      }
      return bars;
    } catch (e) {
      if (attempt < 2) await new Promise(r => setTimeout(r, 2000));
      else throw e;
    }
  }
  throw new Error(`Failed to fetch ${ticker}`);
}

async function main() {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  console.log('Downloading OHLCV data from Yahoo Finance...');

  const allData: Record<string, { dates: string[]; close: number[] }> = {};

  for (const ticker of TICKERS) {
    console.log(`  Fetching ${ticker}...`);
    const bars = await fetchTicker(ticker);
    console.log(`  ${ticker}: ${bars.length} bars`);
    allData[ticker] = { dates: bars.map(b => b.date), close: bars.map(b => b.close) };
    await new Promise(r => setTimeout(r, 500));
  }

  // Save full dataset
  fs.writeFileSync(path.join(DATA_DIR, 'all.json'), JSON.stringify(allData, null, 2));

  // Split train/val
  const trainData: Record<string, { dates: string[]; close: number[] }> = {};
  const valData: Record<string, { dates: string[]; close: number[] }> = {};

  for (const [ticker, data] of Object.entries(allData)) {
    const train: { dates: string[]; close: number[] } = { dates: [], close: [] };
    const val: { dates: string[]; close: number[] } = { dates: [], close: [] };
    for (let i = 0; i < data.dates.length; i++) {
      if (data.dates[i] <= TRAIN_END) { train.dates.push(data.dates[i]); train.close.push(data.close[i]); }
      else if (data.dates[i] <= VAL_END) { val.dates.push(data.dates[i]); val.close.push(data.close[i]); }
    }
    trainData[ticker] = train;
    valData[ticker] = val;
  }

  fs.writeFileSync(path.join(DATA_DIR, 'train.json'), JSON.stringify(trainData, null, 2));
  fs.writeFileSync(path.join(DATA_DIR, 'val.json'), JSON.stringify(valData, null, 2));

  // Generate variant config
  const variants: any[] = [];
  let id = 0;
  const signalTickers = ['SPY', 'QQQ', 'IWM'];
  const rsiPeriods = [7, 10, 12, 14, 18, 21, 28];
  const thresholdPairs = [[25,75],[30,70],[33,67],[35,65],[38,62],[40,60],[42,58],[45,55]];
  const universes = [['SPY'],['SPY','QQQ'],['SPY','QQQ','IWM'],['SPY','IWM'],['QQQ','IWM']];
  const momentumOptions = [false, true];

  for (const ticker of signalTickers) {
    for (const period of rsiPeriods) {
      for (const [longT, shortT] of thresholdPairs) {
        for (const universe of universes) {
          for (const momentum of momentumOptions) {
            if (!universe.includes(ticker)) continue;
            variants.push({
              id: id++,
              signalTicker: ticker,
              rsiPeriod: period,
              longThreshold: longT,
              shortThreshold: shortT,
              universe,
              momentumConfirm: momentum,
              description: `${ticker} RSI(${period}) ${longT}/${shortT} ${momentum ? '+mom' : ''} → ${universe.join('+')}`,
            });
          }
        }
      }
    }
  }

  fs.writeFileSync(path.join(DATA_DIR, 'config.json'), JSON.stringify({ variants, chunkSize: 5 }, null, 2));

  const firstTicker = Object.keys(trainData)[0];
  console.log(`\nTrain bars: ${trainData[firstTicker].dates.length}  (${trainData[firstTicker].dates[0]} → ${trainData[firstTicker].dates[trainData[firstTicker].dates.length - 1]})`);
  console.log(`Val bars:   ${valData[firstTicker].dates.length}  (${valData[firstTicker].dates[0]} → ${valData[firstTicker].dates[valData[firstTicker].dates.length - 1]})`);
  console.log(`Variants:   ${variants.length}`);
  console.log(`\nDone. public/data/ is ready for swarm mode.`);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
