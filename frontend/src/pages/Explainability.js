import React, { useEffect, useState } from 'react';
import { Lightbulb, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import apiService from '../api';

const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#6366f1'];

export default function Explainability() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiService.getFeatureImportance()
      .then(r => setData(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="animate-spin w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full" /></div>;
  }

  const features = data?.features || [];
  const shapFeatures = data?.shap_top_features || [];

  const featureData = features.map((f, i) => ({
    name: f.name,
    importance: f.importance,
    fill: COLORS[i % COLORS.length],
  }));

  const shapData = shapFeatures.map((f, i) => ({
    name: f.name,
    shap_value: f.shap_value,
    fill: COLORS[i % COLORS.length],
  }));

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Explainability</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <Lightbulb className="text-primary-600" size={20} />
            <h3 className="font-semibold text-gray-900">Feature Importance</h3>
          </div>
          {featureData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={featureData} layout="vertical" margin={{ left: 100 }}>
                  <XAxis type="number" fontSize={12} />
                  <YAxis type="category" dataKey="name" fontSize={11} width={100} />
                  <Tooltip />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {featureData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </>
          ) : (
            <p className="text-gray-400 text-center py-12">Train models to see feature importance</p>
          )}
        </div>

        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <Info className="text-purple-600" size={20} />
            <h3 className="font-semibold text-gray-900">SHAP Feature Impact</h3>
          </div>
          {shapData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={shapData} layout="vertical" margin={{ left: 100 }}>
                <XAxis type="number" fontSize={12} />
                <YAxis type="category" dataKey="name" fontSize={11} width={100} />
                <Tooltip />
                <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
                  {shapData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-gray-400 text-center py-12">Train models to see SHAP explanations</p>
          )}
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">Feature Descriptions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { name: 'test_pass_rate', desc: 'Percentage of automated tests that passed. Higher values indicate more correct code.' },
            { name: 'cyclomatic_complexity', desc: 'Measure of code complexity based on control flow. Lower is generally better.' },
            { name: 'num_functions', desc: 'Total number of functions/methods in the submission. Indicates code organization.' },
            { name: 'lines_of_code', desc: 'Total lines of code. Context-dependent metric for code volume.' },
            { name: 'runtime_ms', desc: 'Execution time in milliseconds. Lower values indicate more efficient code.' },
            { name: 'memory_usage_mb', desc: 'Peak memory consumption. Lower is better for resource efficiency.' },
            { name: 'num_failed_tests', desc: 'Number of test cases that failed. Zero is ideal.' },
            { name: 'num_warnings', desc: 'Static analysis warnings. Fewer warnings indicate cleaner code.' },
            { name: 'lint_score', desc: 'Code style compliance score (0-1). Higher values mean better style adherence.' },
            { name: 'documentation_score', desc: 'Documentation quality score (0-1). Measures docstrings and comments.' },
          ].map(f => (
            <div key={f.name} className="p-3 bg-gray-50 rounded-lg">
              <code className="text-sm font-medium text-primary-600">{f.name}</code>
              <p className="text-xs text-gray-600 mt-1">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">What is SHAP?</h3>
        <div className="bg-gray-50 p-4 rounded-lg text-sm text-gray-700 space-y-2">
          <p>
            <strong>SHAP (SHapley Additive exPlanations)</strong> is a game-theoretic approach to explain the output of any machine learning model.
            It connects optimal credit allocation with local explanations using the classic Shapley values from cooperative game theory.
          </p>
          <p>
            Each feature's SHAP value represents its contribution to pushing the prediction from the base value (average model output)
            to the final prediction. Positive SHAP values push the prediction higher, negative values push it lower.
          </p>
          <p>
            This provides <strong>consistent</strong> and <strong>locally accurate</strong> explanations that can be used for:
          </p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>Understanding which features drive individual predictions</li>
            <li>Identifying model biases or unexpected behaviors</li>
            <li>Building trust with end users through transparency</li>
            <li>Debugging model failures and edge cases</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
