import React, { useState } from 'react';
import { Brain, Play, Settings } from 'lucide-react';
import toast from 'react-hot-toast';
import apiService from '../api';

export default function TrainPage() {
  const [config, setConfig] = useState({
    dataset_path: 'data/grading_dataset.csv',
    target_column: 'quality_label',
    model_types: ['random_forest', 'logistic_regression', 'lightgbm', 'xgboost'],
    handle_imbalance: true,
    cv_folds: 5,
  });
  const [training, setTraining] = useState(false);
  const [results, setResults] = useState(null);

  const handleTrain = async () => {
    setTraining(true);
    try {
      const res = await apiService.train(config);
      setResults(res.data);
      toast.success(`Best model: ${res.data.best_model}`);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Training failed');
    } finally {
      setTraining(false);
    }
  };

  const toggleModel = (model) => {
    setConfig(prev => ({
      ...prev,
      model_types: prev.model_types.includes(model)
        ? prev.model_types.filter(m => m !== model)
        : [...prev.model_types, model],
    }));
  };

  const allModels = ['random_forest', 'logistic_regression', 'lightgbm', 'xgboost'];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Train Model</h2>

      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <Settings className="text-primary-600" size={20} />
          <h3 className="font-semibold text-gray-900">Configuration</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Dataset Path</label>
            <input
              type="text"
              className="input-field"
              value={config.dataset_path}
              onChange={(e) => setConfig({ ...config, dataset_path: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Target Column</label>
            <input
              type="text"
              className="input-field"
              value={config.target_column}
              onChange={(e) => setConfig({ ...config, target_column: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">CV Folds</label>
            <input
              type="number"
              className="input-field"
              value={config.cv_folds}
              min={2}
              max={20}
              onChange={(e) => setConfig({ ...config, cv_folds: parseInt(e.target.value) || 5 })}
            />
          </div>
          <div className="flex items-center gap-3 pt-6">
            <input
              type="checkbox"
              id="imbalance"
              checked={config.handle_imbalance}
              onChange={(e) => setConfig({ ...config, handle_imbalance: e.target.checked })}
              className="w-4 h-4 text-primary-600 rounded"
            />
            <label htmlFor="imbalance" className="text-sm font-medium text-gray-700">Handle Class Imbalance</label>
          </div>
        </div>
        <div className="mt-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Models to Train</label>
          <div className="flex flex-wrap gap-2">
            {allModels.map(m => (
              <button
                key={m}
                onClick={() => toggleModel(m)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  config.model_types.includes(m)
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {m.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>
        <button
          onClick={handleTrain}
          disabled={training}
          className="btn-primary mt-6 flex items-center gap-2"
        >
          {training ? (
            <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
          ) : (
            <Play size={16} />
          )}
          {training ? 'Training...' : 'Start Training'}
        </button>
      </div>

      {results && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">
              Training Complete — Best: <span className="text-primary-600">{results.best_model}</span>
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="px-3 py-2 text-left">Model</th>
                    <th className="px-3 py-2 text-left">Accuracy</th>
                    <th className="px-3 py-2 text-left">Precision</th>
                    <th className="px-3 py-2 text-left">Recall</th>
                    <th className="px-3 py-2 text-left">F1</th>
                    <th className="px-3 py-2 text-left">ROC AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results.model_comparison).map(([name, m]) => (
                    <tr key={name} className={`border-t ${name === results.best_model ? 'bg-primary-50' : ''}`}>
                      <td className="px-3 py-2 font-medium capitalize">{name.replace('_', ' ')}</td>
                      <td className="px-3 py-2">{(m.accuracy * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">{(m.precision * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">{(m.recall * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">{(m.f1 * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">{m.roc_auc ? `${(m.roc_auc * 100).toFixed(1)}%` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {results.feature_importance?.length > 0 && (
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-4">Feature Importance</h3>
              <div className="space-y-2">
                {results.feature_importance.map((f, i) => (
                  <div key={f.name} className="flex items-center gap-3">
                    <span className="text-sm text-gray-600 w-40 truncate">{f.name}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                      <div
                        className="h-full bg-primary-500 rounded-full"
                        style={{ width: `${Math.min(f.importance * 100, 100)}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 w-12 text-right">{(f.importance * 100).toFixed(1)}%</span>
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
