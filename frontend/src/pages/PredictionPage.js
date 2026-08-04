import React, { useState } from 'react';
import { Zap, Target } from 'lucide-react';
import toast from 'react-hot-toast';
import apiService from '../api';

const FEATURES = [
  { key: 'test_pass_rate', label: 'Test Pass Rate', type: 'number', min: 0, max: 1, step: 0.01, default: 0.75 },
  { key: 'cyclomatic_complexity', label: 'Cyclomatic Complexity', type: 'number', min: 0, step: 0.1, default: 5 },
  { key: 'num_functions', label: 'Number of Functions', type: 'number', min: 0, step: 1, default: 8 },
  { key: 'lines_of_code', label: 'Lines of Code', type: 'number', min: 0, step: 1, default: 200 },
  { key: 'runtime_ms', label: 'Runtime (ms)', type: 'number', min: 0, step: 1, default: 150 },
  { key: 'memory_usage_mb', label: 'Memory Usage (MB)', type: 'number', min: 0, step: 0.1, default: 40 },
  { key: 'num_failed_tests', label: 'Failed Tests', type: 'number', min: 0, step: 1, default: 0 },
  { key: 'num_warnings', label: 'Warnings', type: 'number', min: 0, step: 1, default: 2 },
  { key: 'lint_score', label: 'Lint Score', type: 'number', min: 0, max: 1, step: 0.01, default: 0.85 },
  { key: 'documentation_score', label: 'Documentation Score', type: 'number', min: 0, max: 1, step: 0.01, default: 0.6 },
];

export default function PredictionPage() {
  const [features, setFeatures] = useState(
    Object.fromEntries(FEATURES.map(f => [f.key, f.default]))
  );
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (key, value) => {
    setFeatures(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await apiService.predictGrading(features);
      setResult(res.data);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const presetProfiles = [
    { label: 'Excellent Submission', values: { test_pass_rate: 0.98, cyclomatic_complexity: 3, num_functions: 12, lines_of_code: 150, runtime_ms: 50, memory_usage_mb: 20, num_failed_tests: 0, num_warnings: 0, lint_score: 0.95, documentation_score: 0.9 } },
    { label: 'Average Submission', values: { test_pass_rate: 0.7, cyclomatic_complexity: 8, num_functions: 5, lines_of_code: 300, runtime_ms: 200, memory_usage_mb: 60, num_failed_tests: 2, num_warnings: 5, lint_score: 0.7, documentation_score: 0.5 } },
    { label: 'Poor Submission', values: { test_pass_rate: 0.3, cyclomatic_complexity: 15, num_functions: 3, lines_of_code: 80, runtime_ms: 500, memory_usage_mb: 120, num_failed_tests: 8, num_warnings: 12, lint_score: 0.3, documentation_score: 0.1 } },
  ];

  const getColor = (label) => {
    if (label === 'excellent') return 'text-green-600 bg-green-50 border-green-200';
    if (label === 'good') return 'text-blue-600 bg-blue-50 border-blue-200';
    if (label === 'needs_improvement') return 'text-amber-600 bg-amber-50 border-amber-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Prediction</h2>

      <div className="card">
        <div className="flex items-center gap-3 mb-4">
          <Zap className="text-primary-600" size={20} />
          <h3 className="font-semibold text-gray-900">Submission Quality Prediction</h3>
        </div>
        <div className="mb-4">
          <p className="text-sm text-gray-500 mb-2">Quick Presets</p>
          <div className="flex flex-wrap gap-2">
            {presetProfiles.map(p => (
              <button key={p.label} onClick={() => setFeatures(p.values)} className="btn-secondary text-xs">
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {FEATURES.map(f => (
            <div key={f.key}>
              <label className="block text-xs font-medium text-gray-600 mb-1">{f.label}</label>
              <input
                type="number"
                className="input-field text-sm"
                value={features[f.key]}
                min={f.min}
                max={f.max}
                step={f.step}
                onChange={(e) => handleChange(f.key, e.target.value)}
              />
            </div>
          ))}
        </div>
        <button onClick={handlePredict} disabled={loading} className="btn-primary mt-4 flex items-center gap-2">
          {loading ? <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" /> : <Target size={16} />}
          {loading ? 'Predicting...' : 'Predict Grade'}
        </button>
      </div>

      {result && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Prediction Result</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className={`p-6 rounded-xl border-2 text-center ${getColor(result.prediction)}`}>
              <p className="text-xs uppercase tracking-wide opacity-70">Predicted Quality</p>
              <p className="text-2xl font-bold mt-1 capitalize">{result.prediction}</p>
            </div>
            <div className="p-6 rounded-xl border border-gray-200 bg-gray-50 text-center">
              <p className="text-xs text-gray-500">Confidence</p>
              <p className="text-3xl font-bold text-primary-600 mt-1">{(result.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="p-6 rounded-xl border border-gray-200 bg-gray-50 text-center">
              <p className="text-xs text-gray-500">Model Used</p>
              <p className="text-lg font-semibold text-gray-800 mt-2 capitalize">{result.model_used?.replace('_', ' ')}</p>
            </div>
          </div>
          {result.probabilities && (
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-700 mb-3">Class Probabilities</h4>
              <div className="space-y-2">
                {Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).map(([label, prob]) => (
                  <div key={label} className="flex items-center gap-3">
                    <span className="text-sm text-gray-600 w-32 capitalize">{label}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-5 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${label === result.prediction ? 'bg-primary-500' : 'bg-gray-300'}`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
