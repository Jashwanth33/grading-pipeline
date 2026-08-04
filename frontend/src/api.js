import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
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
  predictGrading: (data) => api.post('/predict-grading', data),
  predictDoubt: (data) => api.post('/predict-doubt', data),
  getMetrics: () => api.get('/metrics'),
  getFeatureImportance: () => api.get('/feature-importance'),
  getModelInfo: () => api.get('/model-info'),
  uploadDataset: (formData) =>
    api.post('/upload-dataset', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
};

export default apiService;
