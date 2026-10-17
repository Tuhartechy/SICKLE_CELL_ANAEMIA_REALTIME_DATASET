import React, { useState } from 'react';
import { Microscope, ZoomIn, X, ShieldAlert, HeartPulse, Activity } from 'lucide-react';

export const AtlasView: React.FC = () => {
  const [modalImage, setModalImage] = useState<{ src: string; title: string } | null>(null);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
        <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <Microscope className="h-5 w-5 text-rose-600" />
          <span>Pain Scaling & Microvascular Pathology Visual Atlas</span>
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Comparative medical reference linking standardized clinical pain scoring to the underlying cellular pathophysiology of sickle cell vaso-occlusive crisis (VOC).
        </p>
      </div>

      {/* Two-Column Visual Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Figure 1 - Pain Scale */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <div>
                <span className="text-[11px] font-bold text-indigo-600 uppercase tracking-wider">Figure 1</span>
                <h3 className="text-base font-bold text-slate-900">Standardized Clinical Pain Scale (0–10)</h3>
              </div>
              <button
                onClick={() => setModalImage({ src: '/assets/sickle_pain_scale.jpg', title: 'Figure 1: Standardized Clinical Pain Scale' })}
                className="flex items-center gap-1 text-xs font-semibold text-slate-500 hover:text-slate-800 bg-slate-50 hover:bg-slate-100 px-2.5 py-1.5 rounded-lg transition-colors"
              >
                <ZoomIn className="h-3.5 w-3.5" />
                <span>Zoom View</span>
              </button>
            </div>

            {/* Image Container */}
            <div
              onClick={() => setModalImage({ src: '/assets/sickle_pain_scale.jpg', title: 'Figure 1: Standardized Clinical Pain Scale' })}
              className="group relative cursor-pointer overflow-hidden rounded-xl border border-slate-200 bg-slate-50 hover:shadow-md transition-all"
            >
              <img
                src="/assets/sickle_pain_scale.jpg"
                alt="Standardized Pain Severity Scale for Sickle Cell Disease"
                className="w-full h-auto object-cover transition-transform duration-300 group-hover:scale-[1.02]"
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                <span className="opacity-0 group-hover:opacity-100 transition-opacity bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg text-xs font-bold text-slate-900 shadow-md">
                  Click to Expand
                </span>
              </div>
            </div>

            {/* Medical Breakdown */}
            <div className="mt-5 space-y-3.5">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Clinical Severity Stratification
              </h4>

              <div className="p-3 rounded-xl border border-emerald-200 bg-emerald-50/60">
                <h5 className="text-xs font-bold text-emerald-900 flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-emerald-600" />
                  🟢 Mild Pain (Scores 0 – 3)
                </h5>
                <p className="mt-1 text-xs text-emerald-800 leading-relaxed">
                  Manageable dull ache. Vitals stable. Managed with oral hydration (3–4 L/day), oral analgesics (paracetamol / NSAIDs), and monitoring.
                </p>
              </div>

              <div className="p-3 rounded-xl border border-amber-200 bg-amber-50/60">
                <h5 className="text-xs font-bold text-amber-900 flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-amber-600" />
                  🟠 Moderate Pain (Scores 4 – 6)
                </h5>
                <p className="mt-1 text-xs text-amber-800 leading-relaxed">
                  Distress disrupting daily function. Sympathetic activation (HR &gt; 95 bpm). Outpatient observation, oral weak opioids (codeine/tramadol), aggressive hydration.
                </p>
              </div>

              <div className="p-3 rounded-xl border border-rose-200 bg-rose-50/60">
                <h5 className="text-xs font-bold text-rose-900 flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-rose-600" />
                  🔴 Severe Pain (Scores 7 – 10)
                </h5>
                <p className="mt-1 text-xs text-rose-800 leading-relaxed">
                  Excruciating Vaso-Occlusive Crisis (VOC). Severe tachycardia (&gt;115 bpm), tachypnea, facial guarding. Immediate admission, IV opioids within 30 min, IV fluid protocol.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Figure 2 - Cellular Pathology */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <div>
                <span className="text-[11px] font-bold text-rose-600 uppercase tracking-wider">Figure 2</span>
                <h3 className="text-base font-bold text-slate-900">Microvascular Vaso-Occlusion Pathology</h3>
              </div>
              <button
                onClick={() => setModalImage({ src: '/assets/sickle_circulation.jpg', title: 'Figure 2: Microvascular Vaso-Occlusion Pathology' })}
                className="flex items-center gap-1 text-xs font-semibold text-slate-500 hover:text-slate-800 bg-slate-50 hover:bg-slate-100 px-2.5 py-1.5 rounded-lg transition-colors"
              >
                <ZoomIn className="h-3.5 w-3.5" />
                <span>Zoom View</span>
              </button>
            </div>

            {/* Image Container */}
            <div
              onClick={() => setModalImage({ src: '/assets/sickle_circulation.jpg', title: 'Figure 2: Microvascular Vaso-Occlusion Pathology' })}
              className="group relative cursor-pointer overflow-hidden rounded-xl border border-slate-200 bg-slate-50 hover:shadow-md transition-all"
            >
              <img
                src="/assets/sickle_circulation.jpg"
                alt="Normal RBC Laminar Flow vs Sickled RBC Vaso-Occlusion"
                className="w-full h-auto object-cover transition-transform duration-300 group-hover:scale-[1.02]"
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                <span className="opacity-0 group-hover:opacity-100 transition-opacity bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg text-xs font-bold text-slate-900 shadow-md">
                  Click to Expand
                </span>
              </div>
            </div>

            {/* Cellular Walkthrough */}
            <div className="mt-5 space-y-3.5">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Cellular Pathophysiology of Crisis
              </h4>

              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <h5 className="text-xs font-bold text-slate-900">1. Normal Biconcave Erythrocytes (Left)</h5>
                <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                  Healthy RBCs (HbA) are flexible 7.5 µm discs that deform smoothly through 3–5 µm capillaries to deliver oxygen efficiently. Microvascular laminar flow remains unobstructed.
                </p>
              </div>

              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <h5 className="text-xs font-bold text-slate-900">2. HbS Polymerization & Deformation</h5>
                <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                  Under hypoxia, acidosis, dehydration, or cold, mutant Hemoglobin S polymerizes into rigid fiber bundles. RBCs distort into inflexible crescent/sickle shapes with exposed adhesive molecules.
                </p>
              </div>

              <div className="p-3 rounded-xl border border-slate-200 bg-slate-50">
                <h5 className="text-xs font-bold text-slate-900">3. Microvascular Plugging & Ischemic Pain</h5>
                <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                  Sickled cells adhere to endothelium, trapping leukocytes and platelets. Capillaries plug completely, causing acute downstream bone marrow infarction and agonizing nociceptive pain firing.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Multimodal Biomarker Triage Reference Table */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-100">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-bold text-indigo-600 uppercase tracking-wider">Clinical Reference Sheet</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 font-bold border border-indigo-200">
                Multimodal Biomarkers
              </span>
            </div>
            <h3 className="text-base font-bold text-slate-900 mt-1 flex items-center gap-2">
              <Activity className="h-4 w-4 text-rose-600" />
              <span>Physiological & Laboratory Parameters by Crisis Severity Tier</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Comprehensive reference sheet detailing how vital signs, hematology, and organ markers vary from steady-state baseline to life-threatening vaso-occlusive crisis.
            </p>
          </div>

          <div className="inline-flex items-center gap-2 text-xs font-semibold shrink-0">
            <span className="px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 font-bold">
              🟢 Tier 1: Mild
            </span>
            <span className="px-2.5 py-1 rounded-full bg-amber-50 text-amber-800 border border-amber-200 font-bold">
              🟠 Tier 2: Mod
            </span>
            <span className="px-2.5 py-1 rounded-full bg-rose-50 text-rose-800 border border-rose-200 font-bold">
              🔴 Tier 3: Severe
            </span>
          </div>
        </div>

        {/* Responsive Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-200">
          <table className="w-full text-left text-xs text-slate-600 divide-y divide-slate-200">
            <thead className="bg-slate-50 text-[11px] font-bold text-slate-700 uppercase tracking-wider">
              <tr>
                <th className="py-3 px-3.5">Category</th>
                <th className="py-3 px-3.5">Biomarker / Vital</th>
                <th className="py-3 px-3.5 text-emerald-700 bg-emerald-50/50">🟢 Mild (Stable)</th>
                <th className="py-3 px-3.5 text-amber-700 bg-amber-50/50">🟠 Moderate (Guarded)</th>
                <th className="py-3 px-3.5 text-rose-700 bg-rose-50/50">🔴 Severe (Crisis)</th>
                <th className="py-3 px-3.5 text-slate-800">Pathophysiology & Triage Significance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {/* Row 1: Pain Score */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-bold text-slate-700">Clinical Scale</td>
                <td className="py-3 px-3.5 font-semibold text-slate-900">Self-Reported Pain</td>
                <td className="py-3 px-3.5 font-bold text-emerald-700 bg-emerald-50/30">0 – 3 (Manageable)</td>
                <td className="py-3 px-3.5 font-bold text-amber-700 bg-amber-50/30">4 – 6 (Disruptive)</td>
                <td className="py-3 px-3.5 font-extrabold text-rose-700 bg-rose-50/30">7 – 10 (Excruciating)</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Subjective scale from Fig 1; omitted from ML inputs to avoid reporting bias.
                </td>
              </tr>

              {/* Row 2: Heart Rate */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-bold text-slate-700" rowSpan={5}>
                  Autonomic Vital Signs
                </td>
                <td className="py-3 px-3.5 font-semibold text-slate-900">Heart Rate (HR)</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">68 – 78 bpm</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">95 – 104 bpm</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≥ 115 – 130 bpm</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Tachycardia reflects intense sympathetic pain surge and compensatory cardiac output.
                </td>
              </tr>

              {/* Row 3: Respiratory Rate */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-semibold text-slate-900">Respiratory Rate (RR)</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">14 – 18 /min</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">20 – 23 /min</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≥ 24 – 30 /min</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Tachypnea compensatory oxygenation; early warning indicator for Acute Chest Syndrome (ACS).
                </td>
              </tr>

              {/* Row 4: SpO2 */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-semibold text-slate-900">Oxygen Saturation (SpO2)</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">98.0 – 99.0%</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">96.0 – 97.5%</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≤ 90.0 – 94.0%</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Hypoxia directly triggers mutant HbS polymer crystallization and sickled RBC formation.
                </td>
              </tr>

              {/* Row 5: Temperature */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-semibold text-slate-900">Body Temperature</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">98.0 – 98.6°F</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">99.0 – 99.5°F</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≥ 100.4 – 102.0°F</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Fever is the #1 acute crisis precipitation trigger; marks infection or severe bone infarction.
                </td>
              </tr>

              {/* Row 6: Blood Pressure */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-semibold text-slate-900">Blood Pressure (BP)</td>
                <td className="py-3 px-3.5 font-mono text-emerald-700 bg-emerald-50/30">115–120 / 70–75</td>
                <td className="py-3 px-3.5 font-mono text-amber-700 bg-amber-50/30">110–118 / 68–74</td>
                <td className="py-3 px-3.5 font-mono text-rose-700 bg-rose-50/30">100–105 / 60–65</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Hypotension / widened pulse pressure develops in severe pain shock and sepsis risk.
                </td>
              </tr>

              {/* Row 7: Hemoglobin */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-bold text-slate-700" rowSpan={4}>
                  Hematology (CBC)
                </td>
                <td className="py-3 px-3.5 font-semibold text-slate-900">Hemoglobin (Hb)</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">9.0 – 12.6 g/dL</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">7.0 – 8.2 g/dL</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≤ 4.5 – 6.0 g/dL</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Precipitous Hb drop marks active hyperhemolysis or splenic/hepatic blood pooling.
                </td>
              </tr>

              {/* Row 8: WBC */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-semibold text-slate-900">WBC Count</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">4.5k – 6.5k /mcL</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">9.5k – 12k /mcL</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≥ 15k – 22k /mcL</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Neutrophil leukocytosis promotes vessel wall adherence, aggravating vaso-occlusion (Fig 2).
                </td>
              </tr>

              {/* Row 9: Platelets */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-semibold text-slate-900">Platelet Count</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">250k – 320k /mcL</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">170k – 210k /mcL</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≤ 80k – 120k /mcL</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Consumption thrombocytopenia reflects microvascular platelet aggregate trapping.
                </td>
              </tr>

              {/* Row 10: RDW */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-semibold text-slate-900">RDW (%)</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">12.5 – 15.0%</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">19.0 – 22.0%</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≥ 24.0 – 28.0%</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  High anisocytosis reflects accelerated bone marrow reticulocyte release during crisis.
                </td>
              </tr>

              {/* Row 11: Bilirubin */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-bold text-slate-700" rowSpan={2}>
                  Hemolysis & Renal
                </td>
                <td className="py-3 px-3.5 font-semibold text-slate-900">Total Bilirubin</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">0.5 – 1.2 mg/dL</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">2.5 – 4.0 mg/dL</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≥ 5.5 – 8.5 mg/dL</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Unconjugated bilirubin elevation indicates massive intravascular hemolysis and jaundice.
                </td>
              </tr>

              {/* Row 12: Creatinine */}
              <tr className="hover:bg-slate-50/60 transition-colors">
                <td className="py-3 px-3.5 font-semibold text-slate-900">Serum Creatinine</td>
                <td className="py-3 px-3.5 font-mono font-bold text-emerald-700 bg-emerald-50/30">0.50 – 0.65 mg/dL</td>
                <td className="py-3 px-3.5 font-mono font-bold text-amber-700 bg-amber-50/30">0.65 – 0.80 mg/dL</td>
                <td className="py-3 px-3.5 font-mono font-extrabold text-rose-700 bg-rose-50/30">≥ 1.05 – 1.40 mg/dL</td>
                <td className="py-3 px-3.5 text-[11px] text-slate-500">
                  Creatinine spikes from renal medullary hypoperfusion, sickling micro-infarction, and dehydration.
                </td>
              </tr>

              {/* Row 13: Clinical Action */}
              <tr className="bg-slate-50/80 font-semibold">
                <td className="py-3.5 px-3.5 font-bold text-slate-900" colSpan={2}>
                  ⚡ Clinical Triage Action Protocol
                </td>
                <td className="py-3.5 px-3.5 text-emerald-900 bg-emerald-100/60 text-[11px]">
                  <strong>Outpatient Home Protocol:</strong> Oral hydration (3–4 L/day), oral Paracetamol/Ibuprofen, rest & warmth.
                </td>
                <td className="py-3.5 px-3.5 text-amber-900 bg-amber-100/60 text-[11px]">
                  <strong>ED Observation Unit:</strong> 60-min repeat vitals, oral weak opioids (codeine/tramadol), IV hydration.
                </td>
                <td className="py-3.5 px-3.5 text-rose-900 bg-rose-100/60 text-[11px]">
                  <strong>IMMEDIATE RESUSCITATION:</strong> IV opioids within 30 min, IV fluid protocol, continuous pulse oximetry, ACS screen.
                </td>
                <td className="py-3.5 px-3.5 text-slate-700 text-[11px]">
                  Standard clinical decision-support protocol integrated into SickleSense AI predictor.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* High-Resolution Full-Screen Modal */}
      {modalImage && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="relative max-w-4xl w-full bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
            <div className="flex items-center justify-between p-4 border-b border-slate-800 text-white">
              <span className="text-sm font-bold">{modalImage.title}</span>
              <button
                onClick={() => setModalImage(null)}
                className="h-8 w-8 rounded-lg bg-slate-800 hover:bg-slate-700 flex items-center justify-center transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="p-4 bg-black flex items-center justify-center">
              <img
                src={modalImage.src}
                alt={modalImage.title}
                className="max-h-[75vh] w-auto object-contain rounded-lg"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
