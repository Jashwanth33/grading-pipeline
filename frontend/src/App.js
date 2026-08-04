import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import {
  LayoutDashboard, Upload, Brain, BarChart3, HelpCircle,
  Zap, GitBranch, Lightbulb, Menu, X, Activity
} from 'lucide-react';
import Dashboard from './pages/Dashboard';
import UploadPage from './pages/UploadPage';
import TrainPage from './pages/TrainPage';
import EvaluationPage from './pages/EvaluationPage';
import DoubtsPage from './pages/DoubtsPage';
import PredictionPage from './pages/PredictionPage';
import ConfidenceRouting from './pages/ConfidenceRouting';
import Explainability from './pages/Explainability';

const navItems = [
  { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/upload', icon: Upload, label: 'Upload Dataset' },
  { path: '/train', icon: Brain, label: 'Train Model' },
  { path: '/evaluation', icon: BarChart3, label: 'Evaluation' },
  { path: '/doubts', icon: HelpCircle, label: 'Student Doubts' },
  { path: '/prediction', icon: Zap, label: 'Prediction' },
  { path: '/routing', icon: GitBranch, label: 'Confidence Routing' },
  { path: '/explainability', icon: Lightbulb, label: 'Explainability' },
];

function Sidebar({ open, setOpen }) {
  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-md"
      >
        {open ? <X size={20} /> : <Menu size={20} />}
      </button>
      <aside
        className={`fixed top-0 left-0 h-full w-64 bg-white border-r border-gray-200 transform transition-transform duration-200 ease-in-out z-40 ${
          open ? 'translate-x-0' : '-translate-x-full'
        } lg:translate-x-0`}
      >
        <div className="p-6 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary-600 rounded-xl flex items-center justify-center">
              <Activity className="text-white" size={20} />
            </div>
            <div>
              <h1 className="font-bold text-gray-900">ML Pipeline</h1>
              <p className="text-xs text-gray-500">Grading & Triage</p>
            </div>
          </div>
        </div>
        <nav className="p-4 space-y-1">
          {navItems.map(({ path, icon: Icon, label }) => (
            <NavLink
              key={path}
              to={path}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  );
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return (
    <BrowserRouter>
      <Toaster position="top-right" />
      <div className="min-h-screen bg-gray-50">
        <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} />
        <main className="lg:ml-64 min-h-screen">
          <header className="bg-white border-b border-gray-200 px-6 py-4">
            <h2 className="text-lg font-semibold text-gray-800">ML Grading & Doubt Triage Pipeline</h2>
          </header>
          <div className="p-6">
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/train" element={<TrainPage />} />
              <Route path="/evaluation" element={<EvaluationPage />} />
              <Route path="/doubts" element={<DoubtsPage />} />
              <Route path="/prediction" element={<PredictionPage />} />
              <Route path="/routing" element={<ConfidenceRouting />} />
              <Route path="/explainability" element={<Explainability />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}
