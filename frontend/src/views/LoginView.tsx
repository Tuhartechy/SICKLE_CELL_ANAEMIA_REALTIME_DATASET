import React, { useState } from 'react';
import { Activity, ShieldCheck, Lock, Mail, ArrowRight, UserCheck, Stethoscope, Zap, HeartPulse, CheckCircle2 } from 'lucide-react';
import { UserProfile } from '../types';
import { loginUser } from '../api';

interface LoginViewProps {
  onLoginSuccess: (user: UserProfile) => void;
}

export const LoginView: React.FC<LoginViewProps> = ({ onLoginSuccess }) => {
  const [email, setEmail] = useState('dr.sharma@hospital.org');
  const [password, setPassword] = useState('demo123');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await loginUser(email, password);
      onLoginSuccess(data.user);
    } catch (err: any) {
      setError(err.message || 'Login failed. Please check credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickLogin = (demoEmail: string) => {
    setEmail(demoEmail);
    setPassword('demo123');
    setLoading(true);
    setTimeout(async () => {
      try {
        const data = await loginUser(demoEmail, 'demo123');
        onLoginSuccess(data.user);
      } catch {
        // fallback
      } finally {
        setLoading(false);
      }
    }, 350);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 flex flex-col justify-center py-12 sm:px-6 lg:px-8 relative overflow-hidden">
      {/* Ambient Lighting Circles */}
      <div className="absolute top-10 left-1/4 w-96 h-96 bg-rose-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-10 right-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Brand Header */}
      <div className="sm:mx-auto sm:w-full sm:max-w-md relative z-10 text-center">
        <div className="inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-rose-600 to-red-500 text-white shadow-2xl shadow-rose-600/30 mb-4 ring-1 ring-white/20">
          <Activity className="h-9 w-9 stroke-[2.5]" />
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          SickleSense AI
        </h1>
        <p className="mt-2 text-sm text-slate-300 font-medium">
          Clinical Decision Support • Pain Severity & Crisis Risk Stratification
        </p>

        {/* Feature Badges */}
        <div className="mt-3 flex items-center justify-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 px-2.5 py-0.5 text-[11px] font-semibold text-indigo-300">
            <Zap className="h-3 w-3 text-indigo-400" /> 94.9% Triage Accuracy
          </span>
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-300">
            <CheckCircle2 className="h-3 w-3 text-emerald-400" /> 100% Severe Crisis Recall
          </span>
        </div>
      </div>

      <div className="mt-7 sm:mx-auto sm:w-full sm:max-w-md relative z-10 px-4">
        <div className="bg-slate-900/80 border border-slate-800/90 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl">
          {error && (
            <div className="mb-4 rounded-xl bg-rose-500/10 border border-rose-500/30 p-3 text-xs text-rose-300">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Clinical Email Address
              </label>
              <div className="mt-1.5 relative rounded-xl shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Mail className="h-4 w-4" />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="block w-full pl-10 pr-3 py-2.5 bg-slate-950/70 border border-slate-800 rounded-xl text-sm text-white placeholder-slate-500 focus:outline-none focus:border-rose-500 focus:ring-1 focus:ring-rose-500 transition-colors"
                  placeholder="doctor@hospital.org"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Hospital Portal Password
              </label>
              <div className="mt-1.5 relative rounded-xl shadow-xs">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="h-4 w-4" />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="block w-full pl-10 pr-3 py-2.5 bg-slate-950/70 border border-slate-800 rounded-xl text-sm text-white placeholder-slate-500 focus:outline-none focus:border-rose-500 focus:ring-1 focus:ring-rose-500 transition-colors"
                  placeholder="••••••••"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full mt-2 flex items-center justify-center gap-2 py-3 px-4 rounded-xl text-sm font-bold text-white bg-gradient-to-r from-rose-600 via-rose-500 to-red-600 hover:from-rose-500 hover:to-red-500 shadow-lg shadow-rose-600/30 focus:outline-none transition-all disabled:opacity-50"
            >
              {loading ? (
                <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  <span>Authenticate & Launch Triage Portal</span>
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>
          </form>

          {/* Quick Demo Clinician Profiles */}
          <div className="mt-6 pt-5 border-t border-slate-800/80">
            <p className="text-xs font-bold text-slate-400 text-center mb-3">
              ⚡ Quick Clinician Demo Profiles (Instant One-Click Login)
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => handleQuickLogin('dr.sharma@hospital.org')}
                className="flex items-center gap-2.5 p-3 rounded-xl border border-slate-800 bg-slate-950/60 hover:border-rose-500/40 hover:bg-slate-800/60 text-left transition-all group"
              >
                <div className="h-8 w-8 rounded-lg bg-rose-500/10 flex items-center justify-center text-rose-400 group-hover:bg-rose-500/20 shrink-0">
                  <Stethoscope className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-200">Dr. Rohit Sharma</div>
                  <div className="text-[10px] text-slate-400">Senior Hematologist</div>
                </div>
              </button>

              <button
                type="button"
                onClick={() => handleQuickLogin('dr.patel@hospital.org')}
                className="flex items-center gap-2.5 p-3 rounded-xl border border-slate-800 bg-slate-950/60 hover:border-indigo-500/40 hover:bg-slate-800/60 text-left transition-all group"
              >
                <div className="h-8 w-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400 group-hover:bg-indigo-500/20 shrink-0">
                  <UserCheck className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-xs font-bold text-slate-200">Dr. Priya Patel</div>
                  <div className="text-[10px] text-slate-400">ER Triage Specialist</div>
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Security Badge */}
        <div className="mt-6 flex items-center justify-center gap-2 text-xs text-slate-400">
          <ShieldCheck className="h-4 w-4 text-emerald-400" />
          <span>Encrypted Clinical Assessment Protocol • Hospital Grade Triage Suite</span>
        </div>
      </div>
    </div>
  );
};
