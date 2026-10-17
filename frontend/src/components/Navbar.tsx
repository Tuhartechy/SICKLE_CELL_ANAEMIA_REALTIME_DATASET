import React from 'react';
import { Activity, ShieldCheck, User, LogOut, CheckCircle2, Zap } from 'lucide-react';
import { UserProfile } from '../types';

interface NavbarProps {
  user: UserProfile;
  activeModel: 'priority' | 'baseline';
  onModelToggle: (model: 'priority' | 'baseline') => void;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  user,
  activeModel,
  onModelToggle,
  onLogout,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand / Institution */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-rose-600 to-red-500 text-white shadow-md shadow-rose-500/20">
            <Activity className="h-6 w-6 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-bold tracking-tight text-slate-900">
                SickleSense AI
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700 ring-1 ring-inset ring-emerald-600/20">
                <ShieldCheck className="h-3 w-3" /> Clinical Suite
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Hemoglobinopathy Center of Excellence • Clinical Decision Support Suite
            </p>
          </div>
        </div>

        {/* Center / Model Switcher Pill */}
        <div className="hidden md:flex items-center rounded-xl border border-slate-200 bg-slate-100 p-1">
          <button
            onClick={() => onModelToggle('priority')}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
              activeModel === 'priority'
                ? 'bg-white text-indigo-700 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Zap className="h-3.5 w-3.5 fill-indigo-600 text-indigo-600" />
            Augmented Ensemble (96.4%)
          </button>
          <button
            onClick={() => onModelToggle('baseline')}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
              activeModel === 'baseline'
                ? 'bg-white text-slate-800 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <CheckCircle2 className="h-3.5 w-3.5 text-slate-600" />
            Real Dataset (27.0%)
          </button>
        </div>

        {/* User Profile & Logout */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex flex-col text-right">
            <span className="text-xs font-semibold text-slate-900">{user.name}</span>
            <span className="text-[11px] text-slate-500">{user.role}</span>
          </div>
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-100 ring-1 ring-slate-200 text-slate-700">
            <User className="h-4 w-4" />
          </div>
          <button
            onClick={onLogout}
            title="Sign out of clinical session"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition-colors"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
