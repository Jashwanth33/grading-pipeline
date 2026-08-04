import React, { useEffect, useState } from 'react';
import { Route, CheckCircle, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid, Legend } from 'recharts';
import apiService from '../api';

export default function ConfidenceRouting() {
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

  const analysis = metrics?.threshold_analysis || {};
  const results = analysis.results || [];
  const optimalThreshold = analysis.optimal_threshold || 0.75;

  const thresholdData = results.map(r => ({
    threshold: r.threshold,
    f1: r.f1,
    coverage: r.coverage,
    auto_approve: r.auto_approve_count,
    manual_review: r.manual_review_count,
  }));

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Confidence Routing</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card text-center border-2 border-primary-200">
          <Route className="mx-auto text-primary-600 mb-2" size={24} />
          <p className="text-sm text-gray-500">Optimal Threshold</p>
          <p className="text-3xl font-bold text-primary-600">{optimalThreshold.toFixed(2)}</p>
        </div>
        <div className="card text-center">
          <CheckCircle className="mx-auto text-green-500 mb-2" size={24} />
          <p className="text-sm text-gray-500">Auto-Approve Rate</p>
          <p className="text-3xl font-bold text-green-600">
            {analysis.optimal_coverage ? `${(analysis.optimal_coverage * 100).toFixed(1)}%` : '—'}
          </p>
        </div>
        <div className="card text-center">
          <AlertTriangle className="mx-auto text-amber-500 mb-2" size={24} />
          <p className="text-sm text-gray-500">Teacher Review Needed</p>
          <p className="text-3xl font-bold text-amber-600">
            {analysis.optimal_coverage ? `${((1 - analysis.optimal_coverage) * 100).toFixed(1)}%` : '—'}
          </p>
        </div>
      </div>

      {analysis.justification && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-3">Threshold Justification</h3>
          <p className="text-sm text-gray-600 leading-relaxed bg-primary-50 p-4 rounded-lg border border-primary-100">
            {analysis.justification}
          </p>
        </div>
      )}

      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">Threshold vs Metrics</h3>
        {thresholdData.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={thresholdData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="threshold" label={{ value: 'Threshold', position: 'bottom', offset: -5 }} fontSize={12} />
              <YAxis yAxisId="left" domain={[0, 1]} fontSize={12} />
              <YAxis yAxisId="right" orientation="right" fontSize={12} />
              <Tooltip />
              <Legend />
              <ReferenceLine yAxisId="left" x={optimalThreshold} stroke="#3b82f6" strokeDasharray="5 5" label="Optimal" />
              <Line yAxisId="left" type="monotone" dataKey="f1" stroke="#3b82f6" strokeWidth={2} name="F1 Score" dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="coverage" stroke="#22c55e" strokeWidth={2} name="Coverage" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-400 text-center py-12">No threshold analysis data available. Train models first.</p>
        )}
      </div>

      {thresholdData.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Threshold Details</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-3 py-2 text-left">Threshold</th>
                  <th className="px-3 py-2 text-left">F1 Score</th>
                  <th className="px-3 py-2 text-left">Coverage</th>
                  <th className="px-3 py-2 text-left">Auto-Approve</th>
                  <th className="px-3 py-2 text-left">Manual Review</th>
                </tr>
              </thead>
              <tbody>
                {thresholdData.map((r, i) => (
                  <tr key={i} className={`border-t ${r.threshold === optimalThreshold ? 'bg-primary-50 font-medium' : ''}`}>
                    <td className="px-3 py-2">{r.threshold.toFixed(2)} {r.threshold === optimalThreshold && '⭐'}</td>
                    <td className="px-3 py-2">{(r.f1 * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2">{(r.coverage * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 text-green-600">{r.auto_approve}</td>
                    <td className="px-3 py-2 text-amber-600">{r.manual_review}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">How Routing Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-green-50 rounded-xl border border-green-200">
            <CheckCircle className="text-green-600 mb-2" size={20} />
            <h4 className="font-medium text-green-800 mb-2">Auto-Approve</h4>
            <p className="text-sm text-green-700">
              When the model's confidence score is <strong>above the threshold</strong>, the prediction is automatically
              approved without teacher intervention. This saves time on clear-cut cases.
            </p>
          </div>
          <div className="p-4 bg-amber-50 rounded-xl border border-amber-200">
            <AlertTriangle className="text-amber-600 mb-2" size={20} />
            <h4 className="font-medium text-amber-800 mb-2">Teacher Review</h4>
            <p className="text-sm text-amber-700">
              When confidence is <strong>below the threshold</strong>, the doubt is flagged for teacher review.
              This ensures ambiguous cases get human oversight.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
