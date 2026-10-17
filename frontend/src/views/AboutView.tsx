import React from 'react';
import {
  BookOpen,
  ShieldCheck,
  AlertTriangle,
  Layers,
  HeartHandshake,
  CheckCircle2,
  FileCode,
  GraduationCap
} from 'lucide-react';

export const AboutView: React.FC = () => {
  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs">
        <div className="inline-flex items-center gap-2 rounded-full bg-rose-50 px-3 py-1 text-xs font-bold text-rose-700 mb-3 border border-rose-100">
          <GraduationCap className="h-4 w-4" />
          <span>SickleSense AI • Clinical & Technical Documentation</span>
        </div>
        <h1 className="text-2xl font-black text-slate-900">
          Ensemble Machine Learning Approach for Predicting Pain Severity in Sickle Cell Disease
        </h1>
        <p className="mt-2 text-sm text-slate-600 leading-relaxed">
          An investigational clinical decision-support framework trained and validated on hospital triage encounters to predict acute vaso-occlusive crisis severity from objective physiological vitals and hematological markers.
        </p>
      </div>

      {/* Grid of Key Methodological Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Pillar 1: Problem & Objective */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs space-y-3">
          <div className="flex items-center gap-2.5 text-indigo-700">
            <BookOpen className="h-5 w-5" />
            <h3 className="text-sm font-bold uppercase tracking-wider">Clinical Problem & Scope</h3>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Sickle Cell Disease (SCD) is a monogenic hemoglobin disorder where point mutation (Glu6Val) causes abnormal hemoglobin S (HbS) to polymerize under deoxygenation, transforming flexible erythrocytes into rigid sickle-shaped cells.
          </p>
          <p className="text-xs text-slate-600 leading-relaxed">
            These cells occlude capillaries, creating <strong>Vaso-Occlusive Crises (VOC)</strong>. Conventional ER triage relies on patient self-reported pain scores (0–10), which are delayed, subjective, and difficult in pediatric or distressed patients. Our AI pipeline predicts pain severity directly from objective vital signs and blood panels.
          </p>
        </div>

        {/* Pillar 2: Zero Data Leakage */}
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs space-y-3">
          <div className="flex items-center gap-2.5 text-emerald-700">
            <ShieldCheck className="h-5 w-5" />
            <h3 className="text-sm font-bold uppercase tracking-wider">Zero Data Leakage Protocol</h3>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            To prevent the artificial inflation of accuracy common in student machine learning papers:
          </p>
          <ul className="space-y-1.5 text-xs text-slate-600">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>Strict Preprocessing Isolation:</strong> All imputation, scaling, and encoding fit strictly inside training folds.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>Strict Feature Exclusion:</strong> Patient_ID, Pain_Score, and Pain_Severity were never included in feature sets.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>Held-Out Real Test Folds:</strong> In every cross-validation fold, test instances were strictly real hospital patients.</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Dataset & Cohort Breakdown */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <Layers className="h-4 w-4 text-rose-600" />
          <span>Hospital Dataset & Cohort Partitioning ($N = 40 \rightarrow N = 23$)</span>
        </h3>
        <p className="text-xs text-slate-600 leading-relaxed">
          The hospital registry provided 40 raw patient records. To guarantee ground truth integrity, we established deterministic criteria:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="p-3.5 rounded-xl border border-emerald-200 bg-emerald-50/50">
            <div className="font-bold text-emerald-900">Clean Labelled Cohort (N = 23)</div>
            <p className="mt-1 text-emerald-800 text-[11px]">
              8 Mild (34.8%), 7 Moderate (30.4%), 8 Severe (34.8%). Used for all supervised training and cross-validation pipelines.
            </p>
          </div>
          <div className="p-3.5 rounded-xl border border-slate-200 bg-slate-50">
            <div className="font-bold text-slate-900">Audit Review Cohort (N = 17)</div>
            <p className="mt-1 text-slate-600 text-[11px]">
              10 blank records + 7 location/timing notes without pain intensity. Preserved in rows_for_review.csv for clinician audit.
            </p>
          </div>
          <div className="p-3.5 rounded-xl border border-indigo-200 bg-indigo-50/50">
            <div className="font-bold text-indigo-900">Expanded Population (N = 5,000)</div>
            <p className="mt-1 text-indigo-800 text-[11px]">
              Bounded synthetic cohort generated via SMOTE and Gaussian Copula with 0 duplicates and 0 range violations for the 96.4% Augmented Stacking Ensemble.
            </p>
          </div>
        </div>
      </div>

      {/* Study Limitations & Future Clinical Roadmap */}
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <span>Clinical Pilot Study Limitations & Future Roadmap</span>
        </h3>

        <div className="space-y-2.5 text-xs text-slate-600 leading-relaxed">
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <strong>1. Sample Size Constraint (N = 23 Real Patients):</strong> Wide confidence intervals reflect pilot sample constraints. Future work requires expanding to $\ge 500$ prospective multicenter emergency department records.
          </div>
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <strong>2. Age Heterogeneity:</strong> Pediatric cases (down to age 4) and adult cases are evaluated against identical baseline thresholds. Future iterations will normalize vitals using age-adjusted z-scores (Pediatric Early Warning Score, PEWS).
          </div>
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <strong>3. Secondary Lab Missingness:</strong> Serum Creatinine (39.1%) and Total Bilirubin (26.1%) required median imputation due to inconsistent hospital test ordering.
          </div>
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <strong>4. Biomarker Enrichment Roadmap:</strong> Incorporating critical hemolysis markers: Serum Lactate Dehydrogenase (LDH), Free Plasma Hemoglobin, Reticulocyte percentage, and High-Sensitivity C-Reactive Protein (hs-CRP).
          </div>
        </div>
      </div>

      {/* Regulatory Notice */}
      <div className="rounded-2xl border border-amber-300 bg-amber-50 p-4 text-xs text-amber-900">
        <div className="flex items-center gap-2 font-bold mb-1">
          <AlertTriangle className="h-4 w-4 text-amber-700" />
          <span>Investigational Clinical Prototype Notice</span>
        </div>
        <p className="leading-relaxed">
          This web application is an academic research prototype developed for educational, methodological, and exploratory clinical decision support. It is <strong>not</strong> a certified medical device and must <strong>never</strong> supersede the judgment of licensed medical personnel or standard hospital triage protocols.
        </p>
      </div>
    </div>
  );
};
