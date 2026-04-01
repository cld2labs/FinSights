import { useState } from 'react';
import { Upload, FileText, X } from 'lucide-react';

const FileUpload = ({
  onSubmit,
  isLoading,
  acceptedTypes,
  fileType,
  title,
  maxFileSize,
  maxFileSizeBytes = 50 * 1024 * 1024,
  uploadWarning,
  processingNotice,
  onDismissWarning,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);

  const setSelectedFile = (nextFile) => {
    setFile(nextFile);
    onDismissWarning?.();
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      const fileExtension = '.' + droppedFile.name.split('.').pop().toLowerCase();

      if (acceptedTypes.includes(fileExtension)) {
        setSelectedFile(droppedFile);
      }
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
  };

  const submitFile = (ignoreUploadWarnings = false) => {
    if (!file) return;

    const formData = new FormData();
    formData.append('type', fileType);
    formData.append('messages', '');
    formData.append('files', file);
    formData.append('max_tokens', 1024);
    formData.append('language', 'en');
    formData.append('summary_type', 'auto');
    formData.append('stream', 'false');
    formData.append('ignore_upload_warnings', String(ignoreUploadWarnings));

    onSubmit(formData, false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    submitFile(false);
  };

  const isOverLocalSizeLimit = !!file && file.size > maxFileSizeBytes;

  return (
    <div className="card animate-fadeIn">
      <div className="flex items-center mb-4">
        <Upload className="h-6 w-6 text-primary-600 mr-2" />
        <h2 className="text-xl font-semibold text-gray-800">{title}</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div
          className={`file-drop-zone ${dragActive ? 'file-drop-zone-active' : 'file-drop-zone-inactive'}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {!file ? (
            <>
              <Upload className="mx-auto h-16 w-16 text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-700 mb-2">
                Drop your file here or click to browse
              </p>
              <p className="text-sm text-gray-500 mb-2">
                Supported formats: {acceptedTypes.join(', ')}
              </p>
              <p className="text-xs text-gray-400 mb-4">
                Maximum file size: {maxFileSize || '50 MB'}
              </p>
              <input
                type="file"
                id="file-upload"
                className="hidden"
                accept={acceptedTypes.join(',')}
                onChange={handleChange}
                disabled={isLoading}
              />
              <label
                htmlFor="file-upload"
                className="btn-secondary cursor-pointer inline-block"
              >
                Browse Files
              </label>
            </>
          ) : (
            <div className="flex items-center justify-between bg-white p-4 rounded-lg border border-gray-200">
              <div className="flex items-center space-x-3">
                <FileText className="h-8 w-8 text-primary-600" />
                <div>
                  <p className="font-medium text-gray-800">{file.name}</p>
                  <p className="text-sm text-gray-500">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleRemoveFile}
                className="text-gray-400 hover:text-red-500 transition-colors"
                disabled={isLoading}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          )}
        </div>

        {isOverLocalSizeLimit && !uploadWarning && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 p-3">
            <p className="text-sm text-amber-900">
              This file is {(file.size / 1024 / 1024).toFixed(2)} MB. The backend limit is {maxFileSize || '50 MB'}, so you will need to confirm the upload before processing continues.
            </p>
          </div>
        )}

        {uploadWarning && (
          <div className="rounded-lg border border-red-300 bg-red-50 p-4 space-y-3">
            <div>
              <p className="text-sm font-semibold text-red-900">Upload warning</p>
              <p className="text-sm text-red-800">{uploadWarning.message}</p>
            </div>

            <div className="space-y-1 text-xs text-red-900">
              {uploadWarning.file_size_mb ? (
                <p>File size: {uploadWarning.file_size_mb} MB of {uploadWarning.max_file_size_mb} MB allowed.</p>
              ) : null}
              {uploadWarning.page_count ? (
                <p>Pages detected: {uploadWarning.page_count}. Pages that will be processed: {uploadWarning.pages_to_process}.</p>
              ) : null}
            </div>

            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => submitFile(true)}
                disabled={isLoading}
                className="btn-primary"
              >
                Upload Anyway
              </button>
              <button
                type="button"
                onClick={onDismissWarning}
                disabled={isLoading}
                className="btn-secondary"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {processingNotice && !uploadWarning && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 p-3">
            <p className="text-sm font-semibold text-amber-900">Processing notice</p>
            <p className="text-sm text-amber-800">{processingNotice.message}</p>
          </div>
        )}

        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            <span className="font-semibold">📋 Document Limits:</span> Maximum file size is 50 MB, and document is limited to 100 pages.
          </p>
        </div>

        <button
          type="submit"
          disabled={!file || isLoading}
          className="btn-primary w-full"
        >
          {isLoading ? 'Generating Summary...' : 'Generate Summary'}
        </button>
      </form>
    </div>
  );
};

export default FileUpload;
