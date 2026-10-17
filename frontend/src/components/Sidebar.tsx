import React from 'react';
import {
  LayoutDashboard,
  Stethoscope,
  Microscope,
  BarChart3,
  BookOpen,
  Zap,
  Info,
  Database
} from 'lucide-react';

export type ActiveTab = 'dashboard' | 'predictor-synthetic' | 'predictor-real' | 'predictor' | 'atlas' | 'benchmarks' | 'about';

interface SidebarProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  activeModel: 'priority' | 'baseline';
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  activeModel
}) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard Overview', icon: LayoutDashboard, badge: null },
    { id: 'predictor-synthetic', label: 'Synthetic Data Predictor', icon: Zap, badge: '96.4%' },
    { id: 'predictor-real', label: 'Real Data Predictor', icon: Database, badge: '27.0%' },
    { id: 'atlas', label: 'Pain & Pathology Atlas', icon: Microscope, badge: 'Visuals' },
    { id: 'benchmarks', label: 'Model Benchmarks', icon: BarChart3, badge: 'Empirical' },
    { id: 'about', label: 'Study & Methodology', icon: BookOpen, badge: null },
  ];

  return (
    <aside className="w-64 shrink-0 border-r border-slate-200 bg-white min-h-[calc(100vh-4rem)] flex flex-col justify-between p-4">
      <div className="space-y-6">
        <div>
          <p className="px-3 text-[11px] font-bold tracking-wider text-slate-400 uppercase">
            Clinical Modules
          </p>
          <nav className="mt-2 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onTabChange(item.id as ActiveTab)}
                  className={`w-full flex items-center justify-between rounded-xl px-3 py-2.5 text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-rose-50 text-rose-700 shadow-sm font-semibold'
                      : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`h-4 w-4 ${isActive ? 'text-rose-600' : 'text-slate-400'}`} />
                    <span>{item.label}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                        isActive
                          ? 'bg-rose-200 text-rose-800'
                          : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Model Status Card */}
        <div className="rounded-xl border border-slate-200 bg-gradient-to-b from-slate-50 to-white p-3.5 shadow-xs">
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${activeModel === 'priority' ? 'bg-indigo-500 animate-pulse' : 'bg-emerald-500'}`} />
            <span className="text-xs font-bold text-slate-800">
              {activeModel === 'priority' ? 'Optimized Stacking Ensemble' : 'Real Dataset Model'}
            </span>
          </div>
          <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
            {activeModel === 'priority'
              ? 'Trained on 5,000 augmented cases. 96.4% CV accuracy, 100% severe recall.'
              : 'Trained on 23 real hospital patients. 27.0% cross-validation accuracy.'}
          </p>
          <div className="mt-2.5 flex items-center justify-between border-t border-slate-100 pt-2 text-[10px] text-slate-400 font-medium">
            <span>Accuracy</span>
            <span className="font-semibold text-slate-700">
              {activeModel === 'priority' ? '96.40% CV Acc' : '27.00% CV Acc'}
            </span>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="border-t border-slate-100 pt-3">
        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <Info className="h-3.5 w-3.5" />
          <span>Research Prototype v2.0</span>
        </div>
        <p className="mt-1 text-[10px] text-slate-400">
          Non-device investigational triage system.
        </p>
      </div>
    </aside>
  );
};
