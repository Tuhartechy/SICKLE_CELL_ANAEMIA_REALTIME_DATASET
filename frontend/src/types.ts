export interface UserProfile {
  name: string;
  role: string;
  department: string;
  hospital: string;
  license: string;
}

export interface PatientFormState {
  age: number;
  sex: 'F' | 'M';
  genotype: string;
  heart_rate: number;
  respiratory_rate: number;
  spo2: number;
  temperature: number;
  bp_systolic: number;
  bp_diastolic: number;
  hemoglobin: number | null;
  wbc: number | null;
  platelets: number | null;
  rdw: number | null;
  bilirubin_total: number | null;
  creatinine: number | null;
  model_type: 'priority' | 'baseline' | 'augmented' | 'real' | 'both';
}

export interface PhysiologicalAlert {
  type: 'danger' | 'warning' | 'info';
  icon: string;
  label: string;
  clinical: string;
}

export interface TriageDetails {
  tier: string;
  color: 'green' | 'amber' | 'red';
  badge: string;
  pain_scale_range: string;
  description: string;
  actions: string[];
}

export interface SingleModelPrediction {
  predicted_class: 'Mild' | 'Moderate' | 'Severe';
  confidence: number;
  probabilities: {
    Mild: number;
    Moderate: number;
    Severe: number;
  };
  triage: TriageDetails;
  model_used: string;
  model_type: string;
  model_accuracy: number;
  dataset_name?: string;
  dataset_type?: string;
  dataset_proof?: string;
}

export interface PredictionResult {
  predicted_class: 'Mild' | 'Moderate' | 'Severe';
  confidence: number;
  probabilities: {
    Mild: number;
    Moderate: number;
    Severe: number;
  };
  triage: TriageDetails;
  alerts: PhysiologicalAlert[];
  model_used: string;
  model_type: string;
  model_accuracy: number;
  dataset_name?: string;
  dataset_type?: string;
  dataset_proof?: string;
  dual_comparison?: {
    augmented: SingleModelPrediction;
    real: SingleModelPrediction;
  };
}

export interface ModelBenchmark {
  id: string;
  name: string;
  cohort: string;
  accuracy: number;
  accuracy_std: number;
  macro_f1: number;
  mild_f1: number;
  mod_f1: number;
  severe_f1: number;
  severe_recall: number;
  is_priority: boolean;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
  clinical: string;
}

export interface ConfusionMatrixDetails {
  title: string;
  cohort: string;
  model_name: string;
  labels: string[];
  matrix: number[][];
  total_test: number;
  accuracy: number;
  severe_recall: number;
  macro_f1: number;
  image_url: string;
}

export interface BenchmarkData {
  summary: {
    title: string;
    narrative: string;
  };
  models: ModelBenchmark[];
  real_models: ModelBenchmark[];
  augmented_models: ModelBenchmark[];
  confusion_matrix_priority: {
    labels: string[];
    matrix: number[][];
    total_test: number;
    accuracy: number;
    severe_recall?: number;
    macro_f1?: number;
    image_url?: string;
  };
  confusion_matrix_augmented?: ConfusionMatrixDetails;
  confusion_matrix_real?: ConfusionMatrixDetails;
  feature_importance: FeatureImportanceItem[];
}

