import React, { useState, useEffect } from 'react';
import {
  Stethoscope,
  Sparkles,
  Database,
  Activity,
  AlertTriangle,
  RotateCcw,
  CheckCircle2,
  Droplet,
  HeartPulse,
  Thermometer,
  Wind,
  ShieldAlert,
  ArrowRight,
  ClipboardList,
  Columns
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { PatientFormState, PredictionResult } from '../types';
import { predictPainSeverity } from '../api';

const DEFAULT_FORM: PatientFormState = {
  age: 25,
  sex: 'F',
  genotype: 'SS',
  heart_rate: 90,
  respiratory_rate: 20,
  spo2: 98.0,
  temperature: 98.6,
  bp_systolic: 110,
  bp_diastolic: 70,
  hemoglobin: 8.5,
  wbc: 9000.0,
  platelets: 250000.0,
  rdw: 20.0,
  bilirubin_total: 2.0,
  creatinine: 0.55,
  model_type: 'augmented',
};

interface PredictorViewProps {
  activeModel: 'priority' | 'baseline';
  onModelToggle: (model: 'priority' | 'baseline') => void;
  initialEngine?: 'augmented' | 'real' | 'both';
}

export const PredictorView: React.FC<PredictorViewProps> = ({
  activeModel,
  onModelToggle,
  initialEngine = 'augmented',
}) => {
  const [selectedEngine, setSelectedEngine] = useState<'augmented' | 'real' | 'both'>(initialEngine);
  const [form, setForm] = useState<PatientFormState>({
    ...DEFAULT_FORM,
    model_type: initialEngine,
  });
  const [pendingLabs, setPendingLabs] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    if (initialEngine) {
      setSelectedEngine(initialEngine);
      setForm((prev) => ({ ...prev, model_type: initialEngine }));
      if (initialEngine === 'real') {
        onModelToggle('baseline');
      } else {
        onModelToggle('priority');
      }
    }
  }, [initialEngine]);

  // Switch engine
  const handleEngineChange = (engine: 'augmented' | 'real' | 'both') => {
    setSelectedEngine(engine);
    setForm((prev) => ({ ...prev, model_type: engine }));
    if (engine === 'real') {
      onModelToggle('baseline');
    } else {
      onModelToggle('priority');
    }
  };

  const handlePreset = (type: 'severe' | 'moderate' | 'mild' | 'reset') => {
    if (type === 'severe') {
      setForm({
        ...form,
        age: 28,
        sex: 'F',
        genotype: 'SS',
        heart_rate: 126,
        respiratory_rate: 28,
        spo2: 93.0,
        temperature: 101.4,
        bp_systolic: 105,
        bp_diastolic: 65,
        hemoglobin: 5.1,
        wbc: 18900.0,
        platelets: 85000.0,
        rdw: 25.5,
        bilirubin_total: 6.8,
        creatinine: 1.10,
      });
      setPendingLabs(false);
    } else if (type === 'moderate') {
      setForm({
        ...form,
        age: 34,
        sex: 'M',
        genotype: 'SS',
        heart_rate: 102,
        respiratory_rate: 22,
        spo2: 97.0,
        temperature: 99.2,
        bp_systolic: 115,
        bp_diastolic: 72,
        hemoglobin: 7.6,
        wbc: 11200.0,
        platelets: 180000.0,
        rdw: 21.0,
        bilirubin_total: 3.4,
        creatinine: 0.70,
      });
      setPendingLabs(false);
    } else if (type === 'mild') {
      setForm({
        ...form,
        age: 42,
        sex: 'F',
        genotype: 'AS',
        heart_rate: 74,
        respiratory_rate: 16,
        spo2: 99.0,
        temperature: 98.6,
        bp_systolic: 120,
        bp_diastolic: 75,
        hemoglobin: 12.6,
        wbc: 5800.0,
        platelets: 295000.0,
        rdw: 13.8,
        bilirubin_total: 0.8,
        creatinine: 0.60,
      });
      setPendingLabs(false);
    } else {
      setForm({ ...DEFAULT_FORM, model_type: selectedEngine });
      setPendingLabs(false);
      setResult(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const payload: PatientFormState = {
        ...form,
        model_type: selectedEngine,
        hemoglobin: pendingLabs ? null : form.hemoglobin,
        wbc: pendingLabs ? null : form.wbc,
        platelets: pendingLabs ? null : form.platelets,
        rdw: pendingLabs ? null : form.rdw,
        bilirubin_total: pendingLabs ? null : form.bilirubin_total,
        creatinine: pendingLabs ? null : form.creatinine,
      };
      const res = await predictPainSeverity(payload);
      setResult(res);
    } catch (err: any) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Real-time alert chips
  const alerts = [];
  if (form.temperature >= 100.4) alerts.push({ label: `Fever: ${form.temperature}°F`, color: 'bg-rose-100 text-rose-800 border-rose-200' });
  if (form.spo2 < 95.0) alerts.push({ label: `Hypoxia: ${form.spo2}% SpO2`, color: 'bg-rose-100 text-rose-800 border-rose-200' });
  if (form.heart_rate >= 105) alerts.push({ label: `Tachycardia: ${form.heart_rate} bpm`, color: 'bg-amber-100 text-amber-800 border-amber-200' });
  if (form.respiratory_rate >= 24) alerts.push({ label: `Tachypnea: ${form.respiratory_rate} /min`, color: 'bg-amber-100 text-amber-800 border-amber-200' });
  if (!pendingLabs && form.hemoglobin !== null && form.hemoglobin <= 6.0) alerts.push({ label: `Severe Anemia: ${form.hemoglobin} g/dL`, color: 'bg-rose-100 text-rose-800 border-rose-200' });
  if (!pendingLabs && form.wbc !== null && form.wbc >= 13000) alerts.push({ label: `Leukocytosis: ${form.wbc} /mcL`, color: 'bg-amber-100 text-amber-800 border-amber-200' });

  // Chart data for probabilities (Single model)
  const chartData = result ? [
    { name: 'Mild', prob: Math.round(result.probabilities.Mild * 100), fill: '#10b981' },
    { name: 'Moderate', prob: Math.round(result.probabilities.Moderate * 100), fill: '#f59e0b' },
    { name: 'Severe', prob: Math.round(result.probabilities.Severe * 100), fill: '#ef4444' },
  ] : [];

  return (
    <div className="space-y-6">
      {/* Header and Model Switcher Bar */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <Stethoscope className="h-5 w-5 text-rose-600" />
              <span>Live Patient Triage Assessment</span>
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Enter patient vital signs and hematology parameters to evaluate with either Real or Augmented dataset models.
            </p>
          </div>

          {/* Model Selection Switcher: 3 Distinct Modes */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-2">
            <span className="text-xs font-semibold text-slate-500">Evaluation Engine:</span>
            <div className="inline-flex rounded-xl bg-slate-100 p-1 border border-slate-200">
              {/* Synthetic Option */}
              <button
                type="button"
                onClick={() => handleEngineChange('augmented')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  selectedEngine === 'augmented'
                    ? 'bg-white text-indigo-700 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Sparkles className="h-3.5 w-3.5 fill-indigo-600 text-indigo-600" />
                <span>Synthetic Data Predictor (96.4%)</span>
              </button>

              {/* Real Option */}
              <button
                type="button"
                onClick={() => handleEngineChange('real')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  selectedEngine === 'real'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Database className="h-3.5 w-3.5 text-slate-600" />
                <span>Real Data Predictor (27.0%)</span>
              </button>

              {/* Compare Both Option */}
              <button
                type="button"
                onClick={() => handleEngineChange('both')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  selectedEngine === 'both'
                    ? 'bg-white text-rose-700 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Columns className="h-3.5 w-3.5 text-rose-600" />
                <span>Compare Both Side-by-Side</span>
              </button>
            </div>
          </div>
        </div>

        {/* Dataset Provenance & Verification Card */}
        {selectedEngine === 'augmented' && (
          <div className="mt-4 rounded-xl border border-indigo-200 bg-gradient-to-r from-indigo-50/80 via-white to-indigo-50/40 p-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <div className="h-8 w-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0 shadow-xs">
                  <Sparkles className="h-4 w-4" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-indigo-950 uppercase tracking-wider">
                      Live AI Engine: Clinically Bounded Synthetic Cohort
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 text-[10px] font-bold">
                      96.40% CV Accuracy • 100% Severe Recall
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Trained on <strong>N = 5,000 cases</strong> (Gaussian Copula & SMOTE). High-order multi-model stacking overcomes raw data sparsity with zero severe false negatives.
                  </p>
                </div>
              </div>
              <div className="text-right shrink-0 text-xs">
                <span className="font-semibold text-slate-500">Model:</span>{' '}
                <span className="font-bold text-indigo-900">Optimized Stacking Ensemble</span>
              </div>
            </div>
          </div>
        )}

        {selectedEngine === 'real' && (
          <div className="mt-4 rounded-xl border border-amber-200 bg-gradient-to-r from-amber-50/80 via-white to-amber-50/40 p-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <div className="h-8 w-8 rounded-lg bg-amber-600 text-white flex items-center justify-center shrink-0 shadow-xs">
                  <Database className="h-4 w-4" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-amber-950 uppercase tracking-wider">
                      Live AI Engine: Real Observational Hospital Cohort
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-slate-200 text-slate-800 text-[10px] font-bold">
                      27.00% CV Accuracy • Small-Sample Pilot
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Trained directly on <strong>N = 23 un-augmented clinical hospital patients</strong>. Demonstrates continuous vital sign overlap under acute sample sparsity.
                  </p>
                </div>
              </div>
              <div className="text-right shrink-0 text-xs">
                <span className="font-semibold text-slate-500">Model:</span>{' '}
                <span className="font-bold text-slate-900">Stacking Ensemble Baseline</span>
              </div>
            </div>
          </div>
        )}

        {selectedEngine === 'both' && (
          <div className="mt-4 rounded-xl border border-rose-200 bg-gradient-to-r from-rose-50/80 via-white to-indigo-50/40 p-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <div className="h-8 w-8 rounded-lg bg-rose-600 text-white flex items-center justify-center shrink-0 shadow-xs">
                  <Columns className="h-4 w-4" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                      Live Comparative Evaluation: Separated Dataset Models
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-800 text-[10px] font-bold">
                      Comparative Proof Mode
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Simultaneously runs the exact same patient vitals through both models to prove the empirical impact of synthetic dataset learning.
                  </p>
                </div>
              </div>
              <div className="text-right shrink-0 text-xs">
                <span className="font-semibold text-slate-500">Comparison:</span>{' '}
                <span className="font-bold text-rose-700">Synthetic (96.4%) vs Real (27.0%)</span>
              </div>
            </div>
          </div>
        )}

        {/* Quick Presets Bar */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-slate-400 mr-1">Clinical Presets:</span>
          <button
            type="button"
            onClick={() => handlePreset('severe')}
            className="px-3 py-1.5 rounded-xl border border-rose-200 bg-rose-50 text-rose-700 text-xs font-semibold hover:bg-rose-100 transition-all flex items-center gap-1.5"
          >
            <span className="h-2 w-2 rounded-full bg-rose-600" />
            🔴 Severe Crisis Profile
          </button>
          <button
            type="button"
            onClick={() => handlePreset('moderate')}
            className="px-3 py-1.5 rounded-xl border border-amber-200 bg-amber-50 text-amber-700 text-xs font-semibold hover:bg-amber-100 transition-all flex items-center gap-1.5"
          >
            <span className="h-2 w-2 rounded-full bg-amber-600" />
            🟠 Moderate Pain Profile
          </button>
          <button
            type="button"
            onClick={() => handlePreset('mild')}
            className="px-3 py-1.5 rounded-xl border border-emerald-200 bg-emerald-50 text-emerald-700 text-xs font-semibold hover:bg-emerald-100 transition-all flex items-center gap-1.5"
          >
            <span className="h-2 w-2 rounded-full bg-emerald-600" />
            🟢 Stable Baseline Profile
          </button>
          <button
            type="button"
            onClick={() => handlePreset('reset')}
            className="px-3 py-1.5 rounded-xl border border-slate-200 bg-slate-50 text-slate-600 text-xs font-semibold hover:bg-slate-100 transition-all flex items-center gap-1 ml-auto"
          >
            <RotateCcw className="h-3 w-3" />
            Reset Medians
          </button>
        </div>

        {/* Active Physiological Alerts */}
        {alerts.length > 0 && (
          <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap items-center gap-2">
            <span className="text-xs font-bold text-slate-700 flex items-center gap-1">
              <AlertTriangle className="h-3.5 w-3.5 text-rose-500" />
              Physiological Alerts:
            </span>
            {alerts.map((alt, idx) => (
              <span
                key={idx}
                className={`text-[11px] font-bold px-2.5 py-1 rounded-lg border ${alt.color}`}
              >
                {alt.label}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Main Form & Results Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Form Inputs (7 Cols) */}
        <form onSubmit={handleSubmit} className="lg:col-span-7 space-y-4">
          {/* Section 1: Demographics & Genotype */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
            <h3 className="text-sm font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-100 text-indigo-700 text-xs font-extrabold">1</span>
              <span>Demographics & Genotype</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Age (Years)</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={form.age}
                  onChange={(e) => setForm({ ...form, age: Number(e.target.value) })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Biological Sex</label>
                <select
                  value={form.sex}
                  onChange={(e) => setForm({ ...form, sex: e.target.value as 'F' | 'M' })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                >
                  <option value="F">Female (F)</option>
                  <option value="M">Male (M)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Genotype</label>
                <select
                  value={form.genotype}
                  onChange={(e) => setForm({ ...form, genotype: e.target.value })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none font-semibold text-rose-700"
                >
                  <option value="SS">HbSS (Sickle Cell Anemia - Severe)</option>
                  <option value="AS">HbAS (Sickle Cell Trait - Mild)</option>
                  <option value="Negative">HbAA (Normal Control)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Section 2: Physiological Vital Signs */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
            <h3 className="text-sm font-bold text-slate-900 mb-3 flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-rose-100 text-rose-700 text-xs font-extrabold">2</span>
              <span>Physiological Vital Signs (Primary VOC Indicators)</span>
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1 flex items-center gap-1">
                  <HeartPulse className="h-3 w-3 text-rose-500" /> Heart Rate (bpm)
                </label>
                <input
                  type="number"
                  min="40"
                  max="200"
                  value={form.heart_rate}
                  onChange={(e) => setForm({ ...form, heart_rate: Number(e.target.value) })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none font-semibold"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1 flex items-center gap-1">
                  <Activity className="h-3 w-3 text-blue-500" /> SpO2 Saturation (%)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="50"
                  max="100"
                  value={form.spo2}
                  onChange={(e) => setForm({ ...form, spo2: Number(e.target.value) })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none font-semibold"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1 flex items-center gap-1">
                  <Wind className="h-3 w-3 text-cyan-500" /> Resp. Rate (/min)
                </label>
                <input
                  type="number"
                  min="8"
                  max="60"
                  value={form.respiratory_rate}
                  onChange={(e) => setForm({ ...form, respiratory_rate: Number(e.target.value) })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1 flex items-center gap-1">
                  <Thermometer className="h-3 w-3 text-amber-500" /> Temp (°F)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="90"
                  max="108"
                  value={form.temperature}
                  onChange={(e) => setForm({ ...form, temperature: Number(e.target.value) })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Systolic BP (mmHg)</label>
                <input
                  type="number"
                  min="60"
                  max="220"
                  value={form.bp_systolic}
                  onChange={(e) => setForm({ ...form, bp_systolic: Number(e.target.value) })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Diastolic BP (mmHg)</label>
                <input
                  type="number"
                  min="30"
                  max="140"
                  value={form.bp_diastolic}
                  onChange={(e) => setForm({ ...form, bp_diastolic: Number(e.target.value) })}
                  className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Section 3: Hematology & Laboratory Biomarkers */}
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-100 text-emerald-700 text-xs font-extrabold">3</span>
                <span>Hematology & Laboratory Biomarkers</span>
              </h3>
              <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pendingLabs}
                  onChange={(e) => setPendingLabs(e.target.checked)}
                  className="rounded text-indigo-600 focus:ring-indigo-500"
                />
                <span>Simulate Pending Labs</span>
              </label>
            </div>

            {!pendingLabs ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1 flex items-center gap-1">
                    <Droplet className="h-3 w-3 text-red-500" /> Hemoglobin (g/dL)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="2"
                    max="20"
                    value={form.hemoglobin ?? 8.5}
                    onChange={(e) => setForm({ ...form, hemoglobin: Number(e.target.value) })}
                    className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none font-semibold text-rose-700"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1 flex items-center gap-1">
                    <ShieldAlert className="h-3 w-3 text-amber-500" /> WBC Count (/mcL)
                  </label>
                  <input
                    type="number"
                    step="100"
                    min="1000"
                    max="60000"
                    value={form.wbc ?? 9000.0}
                    onChange={(e) => setForm({ ...form, wbc: Number(e.target.value) })}
                    className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">Platelets (/mcL)</label>
                  <input
                    type="number"
                    step="1000"
                    min="10000"
                    max="900000"
                    value={form.platelets ?? 250000.0}
                    onChange={(e) => setForm({ ...form, platelets: Number(e.target.value) })}
                    className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">RDW (%)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="10"
                    max="45"
                    value={form.rdw ?? 20.0}
                    onChange={(e) => setForm({ ...form, rdw: Number(e.target.value) })}
                    className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">Bilirubin (mg/dL)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="35"
                    value={form.bilirubin_total ?? 2.0}
                    onChange={(e) => setForm({ ...form, bilirubin_total: Number(e.target.value) })}
                    className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">Creatinine (mg/dL)</label>
                  <input
                    type="number"
                    step="0.05"
                    min="0.1"
                    max="15"
                    value={form.creatinine ?? 0.55}
                    onChange={(e) => setForm({ ...form, creatinine: Number(e.target.value) })}
                    className="w-full px-3 py-2 text-sm border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none"
                  />
                </div>
              </div>
            ) : (
              <div className="rounded-xl bg-slate-50 border border-slate-200 p-3.5 text-xs text-slate-600">
                ⚡ <strong>Median Imputation Active:</strong> Missing laboratory values will be dynamically imputed using training fold median parameters.
              </div>
            )}
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-3.5 px-4 rounded-xl text-sm font-bold text-white bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 shadow-md shadow-rose-600/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? (
              <div className="h-5 w-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <Stethoscope className="h-4 w-4" />
                <span>
                  {selectedEngine === 'both'
                    ? 'Evaluate & Compare Both Separated Dataset Models'
                    : selectedEngine === 'real'
                    ? 'Evaluate on Real Dataset Model (27.0% Acc • N=23)'
                    : 'Evaluate on Synthetic Data Model (96.4% Acc • N=5,000)'}
                </span>
              </>
            )}
          </button>
        </form>

        {/* Results Column (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {result ? (
            <div className="space-y-4 animate-in fade-in duration-300">
              {/* Case 1: Dual Comparison Mode */}
              {result.dual_comparison ? (
                <div className="space-y-4">
                  {/* Header */}
                  <div className="rounded-2xl border border-indigo-200 bg-gradient-to-r from-indigo-900 to-slate-900 text-white p-4 shadow-sm flex items-center justify-between">
                    <div>
                      <div className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                        <Columns className="h-3.5 w-3.5" />
                        <span>Dual Model Clinical Comparison</span>
                      </div>
                      <p className="text-xs text-slate-300 mt-0.5">
                        Evaluated on exact same 15 patient parameters simultaneously across both separated datasets
                      </p>
                    </div>
                  </div>

                  {/* Side-by-Side Dual Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {/* Real Model Card */}
                    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-xs flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                          <span className="text-[11px] font-bold text-slate-700 uppercase flex items-center gap-1">
                            <Database className="h-3 w-3 text-slate-500" />
                            Real Hospital Dataset
                          </span>
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700">
                            27.0% Acc (N = 23)
                          </span>
                        </div>
                        <div className="mt-3">
                          <span className="text-[11px] font-semibold text-slate-400">Prediction:</span>
                          <div className="text-lg font-black text-slate-900 mt-0.5">
                            {result.dual_comparison.real.triage.tier}
                          </div>
                          <div className="text-xs font-semibold text-slate-600 mt-1">
                            Confidence: <span className="font-bold text-slate-900">{result.dual_comparison.real.confidence}%</span>
                          </div>
                        </div>

                        {/* Probabilities */}
                        <div className="mt-3 space-y-1.5 text-[11px] border-t border-slate-100 pt-2">
                          <div className="flex justify-between text-slate-600">
                            <span>Mild:</span>
                            <span className="font-mono font-bold">{Math.round(result.dual_comparison.real.probabilities.Mild * 100)}%</span>
                          </div>
                          <div className="flex justify-between text-slate-600">
                            <span>Moderate:</span>
                            <span className="font-mono font-bold">{Math.round(result.dual_comparison.real.probabilities.Moderate * 100)}%</span>
                          </div>
                          <div className="flex justify-between text-slate-600">
                            <span>Severe:</span>
                            <span className="font-mono font-bold text-rose-600">{Math.round(result.dual_comparison.real.probabilities.Severe * 100)}%</span>
                          </div>
                        </div>
                      </div>
                      <div className="mt-3 pt-2 border-t border-slate-100 text-[10px] text-slate-400 leading-tight">
                        ⚠️ <em>High parameter variance due to N=18 training sample constraint.</em>
                      </div>
                    </div>

                    {/* Synthetic Model Card */}
                    <div className="rounded-2xl border-2 border-indigo-400/60 bg-indigo-50/40 p-4 shadow-xs flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between pb-2 border-b border-indigo-100">
                          <span className="text-[11px] font-bold text-indigo-900 uppercase flex items-center gap-1">
                            <Sparkles className="h-3 w-3 text-indigo-600" />
                            Synthetic Cohort
                          </span>
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-800">
                            96.4% Acc (N = 5,000)
                          </span>
                        </div>
                        <div className="mt-3">
                          <span className="text-[11px] font-semibold text-indigo-700">Prediction:</span>
                          <div className="text-lg font-black text-indigo-950 mt-0.5">
                            {result.dual_comparison.augmented.triage.tier}
                          </div>
                          <div className="text-xs font-semibold text-indigo-800 mt-1">
                            Confidence: <span className="font-bold text-indigo-950">{result.dual_comparison.augmented.confidence}%</span>
                          </div>
                        </div>

                        {/* Probabilities */}
                        <div className="mt-3 space-y-1.5 text-[11px] border-t border-indigo-100 pt-2">
                          <div className="flex justify-between text-slate-600">
                            <span>Mild:</span>
                            <span className="font-mono font-bold">{Math.round(result.dual_comparison.augmented.probabilities.Mild * 100)}%</span>
                          </div>
                          <div className="flex justify-between text-slate-600">
                            <span>Moderate:</span>
                            <span className="font-mono font-bold">{Math.round(result.dual_comparison.augmented.probabilities.Moderate * 100)}%</span>
                          </div>
                          <div className="flex justify-between text-slate-600">
                            <span>Severe:</span>
                            <span className="font-mono font-bold text-rose-600">{Math.round(result.dual_comparison.augmented.probabilities.Severe * 100)}%</span>
                          </div>
                        </div>
                      </div>
                      <div className="mt-3 pt-2 border-t border-indigo-100 text-[10px] text-emerald-700 font-semibold leading-tight">
                        ✓ <em>Optimized Stacking Ensemble (100% Severe Recall).</em>
                      </div>
                    </div>
                  </div>

                  {/* Clinical Actions */}
                  <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                    <h4 className="text-xs font-bold uppercase tracking-wider mb-2 flex items-center gap-1.5 text-slate-800">
                      <ClipboardList className="h-3.5 w-3.5 text-rose-600" />
                      <span>Recommended Clinical Emergency Protocol:</span>
                    </h4>
                    <ul className="space-y-1.5 text-xs text-slate-700">
                      {result.triage.actions.map((act, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-rose-600 shrink-0 mt-1.5" />
                          <span>{act}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : (
                /* Case 2: Single Model Mode (Either Augmented or Real) */
                <div className="space-y-4">
                  {/* Triage Decision Card */}
                  <div
                    className={`rounded-2xl border p-5 shadow-xs ${
                      result.predicted_class === 'Severe'
                        ? 'border-rose-300 bg-rose-50 text-rose-950'
                        : result.predicted_class === 'Moderate'
                        ? 'border-amber-300 bg-amber-50 text-amber-950'
                        : 'border-emerald-300 bg-emerald-50 text-emerald-950'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold uppercase tracking-wider">
                        {result.triage.badge}
                      </span>
                      <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-white/70">
                        Confidence: {result.confidence}%
                      </span>
                    </div>

                    <div className="mt-3">
                      <h2 className="text-2xl font-black tracking-tight">
                        {result.triage.tier}
                      </h2>
                      <p className="text-xs font-semibold opacity-90 mt-0.5">
                        Pain Scale Equivalent: {result.triage.pain_scale_range}
                      </p>
                      <p className="mt-2 text-xs leading-relaxed opacity-85">
                        {result.triage.description}
                      </p>
                    </div>

                    <div className="mt-4 pt-3 border-t border-black/10">
                      <h4 className="text-xs font-bold uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <ClipboardList className="h-3.5 w-3.5" />
                        <span>Clinical Action Guidance:</span>
                      </h4>
                      <ul className="space-y-1.5 text-xs">
                        {result.triage.actions.map((act, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="h-1.5 w-1.5 rounded-full bg-current shrink-0 mt-1.5" />
                            <span>{act}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Probabilities Distribution */}
                  <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
                      Predicted Probability Distribution
                    </h4>
                    <div className="h-40 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                          <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                          <YAxis type="category" dataKey="name" tick={{ fontSize: 12, fontWeight: 600 }} />
                          <Tooltip formatter={(value: any) => [`${value}%`, 'Confidence']} />
                          <Bar dataKey="prob" radius={[0, 8, 8, 0]}>
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="mt-3 pt-3 border-t border-slate-100 space-y-2 text-xs text-slate-500">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Active Dataset:</span>
                        <span className="font-bold text-slate-800">
                          {result.dataset_name || (selectedEngine === 'augmented' ? 'Synthetic Cohort (N = 5,000 cases)' : 'Real Hospital Cohort (N = 23 patients)')}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Model Architecture:</span>
                        <span className="font-semibold text-slate-800">
                          {result.model_used}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Cross-Validation Accuracy:</span>
                        <span className={`font-bold ${result.model_accuracy > 50 ? 'text-emerald-600' : 'text-slate-800'}`}>
                          {result.model_accuracy}%
                        </span>
                      </div>
                      {result.dataset_proof && (
                        <div className="pt-1.5 border-t border-slate-100 text-[11px] text-slate-600 leading-relaxed">
                          <strong>Proof of Dataset Separation:</strong> {result.dataset_proof}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/60 p-8 text-center h-full flex flex-col items-center justify-center">
              <div className="h-12 w-12 rounded-2xl bg-slate-100 flex items-center justify-center text-slate-400 mb-3">
                <Stethoscope className="h-6 w-6" />
              </div>
              <h3 className="text-sm font-bold text-slate-800">Awaiting Clinical Data</h3>
              <p className="mt-1 text-xs text-slate-500 max-w-xs leading-relaxed">
                Select a clinical scenario preset or enter patient vitals, choose your evaluation engine, then click to calculate pain severity.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
