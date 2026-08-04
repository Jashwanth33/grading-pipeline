import React, { useEffect, useState } from 'react';
import { BarChart3, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, LineChart, Line, Legend } from 'recharts';
import apiService from '../api';

export default function EvaluationPage() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiService.getMetrics()
      .then(r => setMetrics(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="animate-spin w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full" /></div>;
  }

  const grading = metrics?.grading || {};
  const topicComp = Object.entries(metrics?.triage_topic || {}).map(([name, r]) => ({
    model: name.replace('_', ' '),
    accuracy: r.accuracy || 0,
    f1: r.f1 || 0,
  }));

  const radarData = Object.entries(grading).filter(([k]) => !k.startsWith('cv_') && !k.startsWith('std_')).map(([name, m]) => ({
    model: name.replace('_', ' '),
    accuracy: m.accuracy || 0,
    f1: m.f1 || 0,
    precision: m.precision || 0,
    recall: m.recall || 0,
  }));

  const cm = grading.confusion_matrix;
  const cmData = cm ? cm.flatMap((row, i) => row.map((val, j) => ({ actual: `Class ${i}`, predicted: `Class ${j}`, count: val }))) : [];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Evaluation</h2>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <p className="text-sm text-gray-500">Accuracy</p>
          <p className="text-3xl font-bold text-primary-600">{grading.accuracy ? `${(grading.accuracy * 100).toFixed(1)}%` : '—'}</p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-500">Precision</p>
          <p className="text-3xl font-bold text-accent-600">{grading.precision ? `${(grading.precision * 100).toFixed(1)}%` : '—'}</p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-500">Recall</p>
          <p className="text-3xl font-bold text-amber-500">{grading.recall ? `${(grading.recall * 100).toFixed(1)}%` : '—'}</p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-500">F1 Score</p>
          <p className="text-3xl font-bold text-purple-600">{grading.f1 ? `${(grading.f1 * 100).toFixed(1)}%` : '—'}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Model Metrics Radar</h3>
          {radarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="model" fontSize={11} />
                <PolarRadiusAxis angle={30} domain={[0, 1]} fontSize={10} />
                <Radar name="Accuracy" dataKey="accuracy" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
                <Radar name="F1" dataKey="f1" stroke="#22c55e" fill="#22c55e" fillOpacity={0.2} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-gray-400 text-center py-12">Train models first</p>
          )}
        </div>
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">NLP Model Comparison</h3>
          {topicComp.length > 0 ? (
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={topicComp}>
                <XAxis dataKey="model" fontSize={12} />
                <YAxis domain={[0, 1]} fontSize={12} />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" fill="#22c55e" radius={[4, 4, 0, 0]} />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-gray-400 text-center py-12">Train NLP models first</p>
          )}
        </div>
      </div>

      {cm && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Confusion Matrix</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-3 py-2 text-left">Actual \ Predicted</th>
                  {cm[0]?.map((_, j) => (
                    <th key={j} className="px-3 py-2 text-center">Class {j}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cm.map((row, i) => (
                  <tr key={i} className="border-t">
                    <td className="px-3 py-2 font-medium">Class {i}</td>
                    {row.map((val, j) => (
                      <td key={j} className={`px-3 py-2 text-center ${i === j ? 'bg-green-50 font-bold text-green-700' : 'bg-red-50 text-red-600'}`}>
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
