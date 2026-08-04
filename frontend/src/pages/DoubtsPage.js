import React, { useState } from 'react';
import { HelpCircle, Send, CheckCircle, AlertTriangle } from 'lucide-react';
import toast from 'react-hot-toast';
import apiService from '../api';

export default function DoubtsPage() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const res = await apiService.predictDoubt({ question: question.trim() });
      setResult(res.data);
    } catch (err) {
      toast.error('Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const sampleQuestions = [
    "How do I implement a binary search tree in Python?",
    "I'm getting a NullPointerException when I try to read from the database, can someone help?",
    "What's the difference between TCP and UDP?",
    "Can you explain the Big O notation for merge sort?",
    "My React component keeps re-rendering in an infinite loop, why?",
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Student Doubt Triage</h2>

      <div className="card">
        <div className="flex items-center gap-3 mb-4">
          <HelpCircle className="text-primary-600" size={20} />
          <h3 className="font-semibold text-gray-900">Submit a Doubt</h3>
        </div>
        <textarea
          className="input-field min-h-[120px] resize-y"
          placeholder="Enter the student's question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={handleSubmit} disabled={loading || !question.trim()} className="btn-primary mt-3 flex items-center gap-2">
          {loading ? <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" /> : <Send size={16} />}
          {loading ? 'Classifying...' : 'Classify Doubt'}
        </button>
      </div>

      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3">Try Sample Questions</h3>
        <div className="space-y-2">
          {sampleQuestions.map((q, i) => (
            <button
              key={i}
              onClick={() => setQuestion(q)}
              className="w-full text-left px-3 py-2 rounded-lg text-sm text-gray-600 hover:bg-gray-50 border border-gray-100"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {result && (
        <div className="space-y-4">
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Classification Result</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500 mb-1">Predicted Topic</p>
                <p className="text-lg font-bold text-primary-600">{result.predicted_topic}</p>
                <p className="text-xs text-gray-400">Confidence: {(result.topic_confidence * 100).toFixed(1)}%</p>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500 mb-1">Predicted Urgency</p>
                <div className="flex items-center gap-2">
                  <p className="text-lg font-bold text-primary-600">{result.predicted_urgency}</p>
                  {result.auto_approve ? (
                    <span className="badge badge-green"><CheckCircle size={12} className="mr-1" />Auto-Approve</span>
                  ) : (
                    <span className="badge badge-red"><AlertTriangle size={12} className="mr-1" />Teacher Review</span>
                  )}
                </div>
                <p className="text-xs text-gray-400">Confidence: {(result.urgency_confidence * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>

          {result.topic_probabilities && (
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-3">Topic Probabilities</h3>
              <div className="space-y-2">
                {Object.entries(result.topic_probabilities).sort((a, b) => b[1] - a[1]).map(([topic, prob]) => (
                  <div key={topic} className="flex items-center gap-3">
                    <span className="text-sm text-gray-600 w-32">{topic}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                      <div className="h-full bg-primary-500 rounded-full" style={{ width: `${prob * 100}%` }} />
                    </div>
                    <span className="text-xs text-gray-400 w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.urgency_probabilities && (
            <div className="card">
              <h3 className="font-semibold text-gray-900 mb-3">Urgency Probabilities</h3>
              <div className="space-y-2">
                {Object.entries(result.urgency_probabilities).sort((a, b) => b[1] - a[1]).map(([urg, prob]) => (
                  <div key={urg} className="flex items-center gap-3">
                    <span className="text-sm text-gray-600 w-32">{urg}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                      <div className="h-full bg-amber-500 rounded-full" style={{ width: `${prob * 100}%` }} />
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
