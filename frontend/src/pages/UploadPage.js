import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import apiService from '../api';

export default function UploadPage() {
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);

  const onDrop = useCallback(async (files) => {
    if (files.length === 0) return;
    const file = files[0];
    const formData = new FormData();
    formData.append('file', file);
    setUploading(true);
    try {
      const res = await apiService.uploadDataset(formData);
      setUploadResult(res.data);
      toast.success(`Uploaded ${res.data.rows} rows successfully`);
    } catch (err) {
      toast.error('Upload failed');
    } finally {
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] },
    maxFiles: 1,
  });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Upload Dataset</h2>
      <div
        {...getRootProps()}
        className={`card cursor-pointer border-2 border-dashed transition-colors ${
          isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'
        }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center py-12">
          <Upload className="w-12 h-12 text-gray-400 mb-4" />
          <p className="text-lg font-medium text-gray-700">
            {isDragActive ? 'Drop your file here' : 'Drag & drop a CSV or JSON file'}
          </p>
          <p className="text-sm text-gray-400 mt-2">or click to browse</p>
        </div>
      </div>
      {uploading && (
        <div className="flex items-center gap-3 text-primary-600">
          <div className="animate-spin w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full" />
          <span>Uploading...</span>
        </div>
      )}
      {uploadResult && (
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle className="text-accent-600" size={20} />
            <h3 className="font-semibold text-gray-900">Upload Successful</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <p className="text-2xl font-bold text-primary-600">{uploadResult.rows}</p>
              <p className="text-xs text-gray-500">Rows</p>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <p className="text-2xl font-bold text-primary-600">{uploadResult.columns}</p>
              <p className="text-xs text-gray-500">Columns</p>
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-2">Columns: {uploadResult.column_names?.join(', ')}</p>
          {uploadResult.preview && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    {uploadResult.column_names?.slice(0, 6).map((col) => (
                      <th key={col} className="px-3 py-2 text-left font-medium text-gray-600">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {uploadResult.preview.slice(0, 5).map((row, i) => (
                    <tr key={i} className="border-t border-gray-100">
                      {uploadResult.column_names?.slice(0, 6).map((col) => (
                        <td key={col} className="px-3 py-2 text-gray-700">
                          {typeof row[col] === 'number' ? row[col]?.toFixed?.(2) ?? row[col] : row[col]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
