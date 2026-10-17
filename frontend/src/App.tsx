import React, { useState } from 'react';
import { UserProfile } from './types';
import { Navbar } from './components/Navbar';
import { Sidebar, ActiveTab } from './components/Sidebar';
import { LoginView } from './views/LoginView';
import { DashboardView } from './views/DashboardView';
import { PredictorView } from './views/PredictorView';
import { AtlasView } from './views/AtlasView';
import { BenchmarksView } from './views/BenchmarksView';
import { AboutView } from './views/AboutView';

export function App() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');
  const [activeModel, setActiveModel] = useState<'priority' | 'baseline'>('priority');

  const handleLogout = () => {
    setUser(null);
  };

  const handleLoginSuccess = (loggedInUser: UserProfile) => {
    setUser(loggedInUser);
    setActiveTab('dashboard');
  };

  if (!user) {
    return <LoginView onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col text-slate-900 font-sans">
      {/* Top Navbar */}
      <Navbar
        user={user}
        activeModel={activeModel}
        onModelToggle={setActiveModel}
        onLogout={handleLogout}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex w-full">
        {/* Navigation Sidebar */}
        <div className="hidden md:block">
          <Sidebar
            activeTab={activeTab}
            onTabChange={setActiveTab}
            activeModel={activeModel}
          />
        </div>

        {/* Mobile View Navigation Bar */}
        <div className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-white border-t border-slate-200 px-3 py-2 flex justify-around">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`text-xs font-semibold px-2 py-1 rounded-lg ${activeTab === 'dashboard' ? 'text-rose-600 bg-rose-50' : 'text-slate-600'}`}
          >
            Dashboard
          </button>
          <button
            onClick={() => setActiveTab('predictor-synthetic')}
            className={`text-xs font-semibold px-2 py-1 rounded-lg ${activeTab === 'predictor-synthetic' || activeTab === 'predictor' ? 'text-indigo-600 bg-indigo-50' : 'text-slate-600'}`}
          >
            Synthetic AI
          </button>
          <button
            onClick={() => setActiveTab('predictor-real')}
            className={`text-xs font-semibold px-2 py-1 rounded-lg ${activeTab === 'predictor-real' ? 'text-amber-600 bg-amber-50' : 'text-slate-600'}`}
          >
            Real Pilot AI
          </button>
          <button
            onClick={() => setActiveTab('atlas')}
            className={`text-xs font-semibold px-2 py-1 rounded-lg ${activeTab === 'atlas' ? 'text-rose-600 bg-rose-50' : 'text-slate-600'}`}
          >
            Atlas
          </button>
          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`text-xs font-semibold px-2 py-1 rounded-lg ${activeTab === 'benchmarks' ? 'text-rose-600 bg-rose-50' : 'text-slate-600'}`}
          >
            Benchmarks
          </button>
        </div>

        {/* View Viewport */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto w-full pb-16 md:pb-8">
          {activeTab === 'dashboard' && (
            <DashboardView
              onNavigate={setActiveTab}
              activeModel={activeModel}
            />
          )}

          {(activeTab === 'predictor' || activeTab === 'predictor-synthetic') && (
            <PredictorView
              initialEngine="augmented"
              activeModel={activeModel}
              onModelToggle={setActiveModel}
            />
          )}

          {activeTab === 'predictor-real' && (
            <PredictorView
              initialEngine="real"
              activeModel={activeModel}
              onModelToggle={setActiveModel}
            />
          )}

          {activeTab === 'atlas' && <AtlasView />}

          {activeTab === 'benchmarks' && <BenchmarksView />}

          {activeTab === 'about' && <AboutView />}
        </main>
      </div>
    </div>
  );
}

export default App;
