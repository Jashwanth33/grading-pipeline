import React, { useEffect, useState } from 'react';
import { BarChart3, Brain, HelpCircle, TrendingUp, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import apiService from '../api';

const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6'];

function StatCard({ icon: Icon, label, value, color, sub }) {
  return (
    <div className="card flex items-start gap-4">
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${color}`}>
        <Icon className="text-white" size={22} />
      </div>
      <div>
        <p className="text-sm text-gray-500">{label}</p>
        <p className="text-2xl font-bold text-gray-900">{value}</p>
        {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiService.getMetrics()
      .then(r => setMetrics(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const gradingMetrics = metrics?.grading || {};
  const topicResults = metrics?.triage_topic || {};
  const urgencyResults = metrics?.triage_urgency || {};

  const modelComparison = Object.entries(gradingMetrics).filter(([k]) => !k.startsWith('cv_')).map(([name, m]) => ({
    name: name.replace('_', ' '),
    accuracy: m.accuracy || 0,
    f1: m.f1 || 0,
  }));

  const topicComp = Object.entries(topicResults).map(([name, r]) => ({
    name: name.replace('_', ' '),
    f1: r.f1 || 0,
  }));

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Brain} label="Best Model" value={gradingMetrics.accuracy ? 'Trained' : 'Not Trained'} color="bg-primary-600" sub={gradingMetrics.f1 ? `F1: ${(gradingMetrics.f1 * 100).toFixed(1)}%` : ''} />
        <StatCard icon={BarChart3} label="Accuracy" value={gradingMetrics.accuracy ? `${(gradingMetrics.accuracy * 100).toFixed(1)}%` : '—'} color="bg-accent-600" />
        <StatCard icon={HelpCircle} label="Topic Models" value={topicComp.length > 0 ? `${topicComp.length} trained` : 'Not trained'} color="bg-amber-500" />
        <StatCard icon={TrendingUp} label="System Status" value="Ready" color="bg-purple-600" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Model Comparison</h3>
          {modelComparison.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparison}>
                <XAxis dataKey="name" fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" fill="#22c55e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-gray-400 text-center py-12">No models trained yet. Go to Train Model page.</p>
          )}
        </div>
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">NLP Model Comparison</h3>
          {topicComp.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={topicComp}>
                <XAxis dataKey="name" fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip />
                <Bar dataKey="f1" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-gray-400 text-center py-12">Train NLP models to see comparison.</p>
          )}
        </div>
      </div>
    </div>
  );
}
