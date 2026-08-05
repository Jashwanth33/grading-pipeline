import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

api.interceptors.request.use((config) => {
  if (config.data instanceof FormData) {
    delete config.headers['Content-Type'];
  } else {
    config.headers['Content-Type'] = 'application/json';
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'Request failed';
    console.error('API Error:', message);
    return Promise.reject(error);
  }
);

export const apiService = {
  train: (data) => api.post('/train', data),
  trainTriage: (data) => api.post('/train-triage', data),
  predictGrading: (data) => api.post('/predict-grading', data),
  predictDoubt: (data) => api.post('/predict-doubt', data),
  getMetrics: () => api.get('/metrics'),
  getFeatureImportance: () => api.get('/feature-importance'),
  getModelInfo: () => api.get('/model-info'),
  uploadDataset: (formData) =>
    api.post('/upload-dataset', formData),
};

export default apiService;
