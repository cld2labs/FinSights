// src/pages/Generate.jsx
import { useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'react-hot-toast';
import { jsPDF } from 'jspdf';

import TextInput from '../components/TextInput';
import FileUpload from '../components/FileUpload';
import { generateSummaryJson, cloneFormData, getRagStatus, ragChat, deleteVectors } from '../services/api';

// Helper function for streaming text effect
const streamText = (text, onUpdate, speedMs = 20) => {
  let index = 0;
  let currentText = '';
  
  const interval = setInterval(() => {
    if (index < text.length) {
      currentText += text[index];
      onUpdate(currentText);
      index++;
    } else {
      clearInterval(interval);
    }
  }, speedMs);
  
  return () => clearInterval(interval);
};

export const Generate = () => {
  const [activeTab, setActiveTab] = useState('text');

  // Store last submitted input (for text flow only)
  const [lastFormData, setLastFormData] = useState(null);

  // Store doc_id returned by backend (for file flow)
  const [docId, setDocId] = useState('');

  // Dynamic section chips returned by backend (2-5)
  const [dynamicSections, setDynamicSections] = useState([]);

  // Chat-like history (summary + section summaries)
  const [history, setHistory] = useState([]);

  const [isLoadingInitial, setIsLoadingInitial] = useState(false);
  const [activeChipLoading, setActiveChipLoading] = useState('');

  // RAG UI state
  const [ragReady, setRagReady] = useState(false);
  const [ragChecking, setRagChecking] = useState(false);
  const [ragQuestion, setRagQuestion] = useState('');
  const [ragLoading, setRagLoading] = useState(false);
  const [chatFocused, setChatFocused] = useState(false);

  const bottomRef = useRef(null);

  const tabs = [
    { id: 'text', label: 'Paste Text' },
    { id: 'file', label: 'Upload File' },
  ];

  // Fallback chips if backend does not return sections
  const fallbackSectionOptions = useMemo(() => ['General Summary', 'Key Extracts'], []);

  const sectionOptions = useMemo(() => {
    if (Array.isArray(dynamicSections) && dynamicSections.length >= 2) return dynamicSections;
    return fallbackSectionOptions;
  }, [dynamicSections, fallbackSectionOptions]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [history, isLoadingInitial, activeChipLoading, dynamicSections, ragLoading, ragReady]);

  // ----------------------------
  // Frontend-only: remove duplicated section heading from LLM output
  // ----------------------------
  const stripLeadingHeading = (text, heading) => {
    if (!text) return '';
    if (!heading) return text.trim();

    const t = String(text).replace(/\r\n/g, '\n').trimStart();
    const h = String(heading).trim();

    const escaped = h.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(`^${escaped}\\s*[:\\-–]?\\s*\\n?`, 'i');
    return t.replace(re, '').trim();
  };

  // ----------------------------
  // PDF Download helpers
  // ----------------------------
  const sanitizeFilename = (name) =>
    (name || 'finsights-summary')
      .toLowerCase()
      .replace(/[^a-z0-9-_]+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '');

  const buildPdfFromHistory = (items) => {
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const marginX = 48;
    const marginTop = 56;
    const marginBottom = 56;
    const contentWidth = pageWidth - marginX * 2;

    let y = marginTop;

    const ensureSpace = (neededHeight) => {
      if (y + neededHeight > pageHeight - marginBottom) {
        doc.addPage();
        y = marginTop;
      }
    };

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(22);
    doc.text('FinSights', marginX, y);
    y += 18;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, marginX, y);
    y += 22;

    const outputs = (items || []).filter(
      (it) => it && it.role === 'assistant' && (it.text || '').trim().length > 0
    );

    if (outputs.length === 0) {
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(12);
      doc.text('No generated outputs to download yet.', marginX, y);
      return doc;
    }

    outputs.forEach((item, idx) => {
      const title = (item.title || `Section ${idx + 1}`).trim();
      const body = stripLeadingHeading(item.text || '', title);

      ensureSpace(24);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(14);
      doc.text(title, marginX, y);
      y += 16;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(11);

      const paragraphs = (body || '').split('\n');
      paragraphs.forEach((p, pIdx) => {
        const lineText = (p || '').trimEnd();
        const lines = doc.splitTextToSize(lineText, contentWidth);

        if (lines.length === 0) {
          ensureSpace(14);
          y += 12;
          return;
        }

        lines.forEach((ln) => {
          ensureSpace(14);
          doc.text(ln, marginX, y);
          y += 14;
        });

        if (pIdx !== paragraphs.length - 1) {
          ensureSpace(10);
          y += 6;
        }
      });

      y += 18;
    });

    return doc;
  };

  const handleDownloadHistory = () => {
    if (!history || history.length === 0) {
      toast.error('No history to download yet.');
      return;
    }

    const doc = buildPdfFromHistory(history);

    const firstTitle =
      history.find((h) => h?.role === 'assistant' && h?.title)?.title || 'summary';
    const filename = `${sanitizeFilename(`finsights-${firstTitle}`)}-${Date.now()}.pdf`;

    doc.save(filename);
  };

  const resetRunState = () => {
    setHistory([]);
    setDocId('');
    setDynamicSections([]);
    setLastFormData(null);

    // RAG
    setRagReady(false);
    setRagChecking(false);
    setRagQuestion('');
    setRagLoading(false);
  };

  const refreshRagStatus = async (id) => {
    const cleanId = (id || '').trim();
    if (!cleanId) {
      setRagReady(false);
      return;
    }

    setRagChecking(true);
    try {
      const status = await getRagStatus(cleanId);
      setRagReady(!!status?.ready);
    } catch (e) {
      console.error('RAG status error:', e);
      setRagReady(false);
    } finally {
      setRagChecking(false);
    }
  };

  const handleSubmit = async (formData) => {
    setIsLoadingInitial(true);
    
    // Clean up old vectors before submitting new document
    if (docId) {
      await deleteVectors(docId);
    }
    
    resetRunState();

    // Store lastFormData only for text flow
    if (activeTab === 'text') setLastFormData(formData);

    try {
      const fd = cloneFormData(formData);
      fd.set('mode', 'financial_initial');

      const json = await generateSummaryJson(fd);

      if (json.doc_id) {
        setDocId(json.doc_id);
        // kick off RAG status check (best-effort)
        refreshRagStatus(json.doc_id);
      }

      // dynamic sections from backend (expected field)
      // Accept either: json.sections = ["A","B"] OR json.sections = [{title:"A"},...]
      const secs = json.sections || json.section_options || [];
      let titles = [];

      if (Array.isArray(secs)) {
        titles = secs
          .map((s) => (typeof s === 'string' ? s : s?.title))
          .filter(Boolean)
          .map((s) => String(s).trim())
          .filter((s) => s.length > 0);
      }

      // enforce 2 to 5 on frontend too (safety)
      if (titles.length >= 2) setDynamicSections(titles.slice(0, 5));
      else setDynamicSections([]);

      const rawText = json.text || json.output_text || '';
      const text = stripLeadingHeading(rawText, 'Generalized Summary');

      // Add response with streaming effect
      setHistory([
        {
          role: 'assistant',
          title: 'Generalized Summary',
          text: '', // Start with empty text
          ts: Date.now(),
        },
      ]);

      // Stream the text
      streamText(text, (streamedText) => {
        setHistory((prev) => {
          const updated = [...prev];
          if (updated.length > 0) {
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              text: streamedText,
            };
          }
          return updated;
        });
      }, 15);

      toast.success('Initial summary generated. Select a section below.');
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to generate summary. Please try again.');
      resetRunState();
    } finally {
      setIsLoadingInitial(false);
    }
  };

  const handleChipClick = async (section) => {
    if (!docId && !lastFormData) {
      toast.error('Please upload a document or paste text first.');
      return;
    }
    if (activeChipLoading || isLoadingInitial || ragLoading) return;
    setChatFocused(false); // Ensure chat is unfocused when clicking section

    setHistory((prev) => [...prev, { role: 'user', text: section, ts: Date.now() }]);
    setActiveChipLoading(section);

    try {
      let fd;

      if (docId) {
        fd = new FormData();
        fd.set('type', 'text');
        fd.set('doc_id', docId);
        fd.set('mode', 'financial_section');
        fd.set('section', section);
      } else {
        fd = cloneFormData(lastFormData);
        fd.set('mode', 'financial_section');
        fd.set('section', section);
      }

      const json = await generateSummaryJson(fd);
      const rawText = json.text || json.output_text || '';
      const text = stripLeadingHeading(rawText, section);

      // Add response with streaming effect
      setHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          title: section,
          text: '', // Start with empty text
          ts: Date.now(),
        },
      ]);

      // Stream the text
      const lastIndex = setHistory.length - 1;
      streamText(text, (streamedText) => {
        setHistory((prev) => {
          const updated = [...prev];
          if (updated.length > 0) {
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              text: streamedText,
            };
          }
          return updated;
        });
      }, 15);

      toast.success(`${section} generated successfully!`);
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to generate section. Please try again.');

      setHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          title: section,
          text: 'Failed to generate this section. Please try again.',
          ts: Date.now(),
        },
      ]);
    } finally {
      setActiveChipLoading('');
    }
  };

  const handleAskRag = async (e) => {
    e?.preventDefault?.();

    const q = (ragQuestion || '').trim();
    if (!docId) {
      toast.error('Upload a document or paste text first.');
      return;
    }
    if (!q) {
      toast.error('Enter a question.');
      return;
    }
    if (isLoadingInitial || activeChipLoading || ragLoading) return;

    setHistory((prev) => [...prev, { role: 'user', text: q, ts: Date.now() }]);
    setRagLoading(true);

    try {
      const resp = await ragChat({ docId, message: q });
      const answer = resp?.text || resp?.answer || resp?.output_text || 'No answer returned.';

      // Add response with streaming effect
      setHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          title: 'Document Q&A',
          text: '', // Start with empty text
          ts: Date.now(),
        },
      ]);

      // Stream the text
      streamText(String(answer), (streamedText) => {
        setHistory((prev) => {
          const updated = [...prev];
          if (updated.length > 0) {
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              text: streamedText,
            };
          }
          return updated;
        });
      }, 15);

      setRagQuestion('');
      setChatFocused(false); // Show sections again after answer
    } catch (err) {
      console.error('RAG chat error:', err);
      toast.error('Failed to answer. Please try again.');

      setHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          title: 'Document Q&A',
          text: 'Failed to answer. Please try again.',
          ts: Date.now(),
        },
      ]);
    } finally {
      setRagLoading(false);
    }
  };

  const renderChat = () => {
    if (isLoadingInitial && history.length === 0) {
      return (
        <div className="flex items-center justify-start py-12 px-4">
          <div className="max-w-[85%] rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 px-5 py-4 text-gray-900 shadow-sm ring-1 ring-primary-100">
            <div className="text-base font-semibold text-gray-800 mb-3">Analyzing Document</div>
            <div className="flex items-center gap-2">
              <div className="text-sm text-gray-600">Gathering information and generating summary</div>
              <div className="flex gap-1">
                <span className="inline-block w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                <span className="inline-block w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                <span className="inline-block w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
              </div>
            </div>
          </div>
        </div>
      );
    }

    if (!isLoadingInitial && history.length === 0) {
      return (
        <div className="text-center py-12">
          <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          </div>
          <p className="text-gray-500">Your summaries will appear here once generated</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {history.map((item, idx) => {
          const key = `${item.ts}-${idx}`;
          const isUser = item.role === 'user';

          if (isUser) {
            return (
              <div key={key} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl bg-primary-600 px-4 py-2 text-white shadow-sm">
                  <div className="text-sm font-medium">Selected</div>
                  <div className="whitespace-pre-wrap">{item.text}</div>
                </div>
              </div>
            );
          }

          return (
            <div key={key} className="flex justify-start">
              <div className="max-w-[85%] rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 px-5 py-4 text-gray-900 shadow-sm ring-1 ring-primary-100">
                {item.title && (
                  <div className="text-base font-semibold text-gray-800 mb-2">{item.title}</div>
                )}
                <div className="whitespace-pre-wrap leading-relaxed">{item.text}</div>
              </div>
            </div>
          );
        })}

        {/* Chips */}
        {history.length > 0 && (
          <div
            className={`pt-2 transition-all duration-500 ease-in-out ${
              chatFocused ? 'opacity-0 max-h-0 overflow-hidden' : 'opacity-100 max-h-96'
            }`}
          >
            <div className="flex flex-wrap items-center justify-center gap-3">
              {sectionOptions.map((label) => {
                const loading = activeChipLoading === label;
                const disabled = isLoadingInitial || !!activeChipLoading || ragLoading;

                return (
                  <button
                    key={label}
                    onClick={() => handleChipClick(label)}
                    disabled={disabled}
                    className={[
                      'inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-medium shadow-sm ring-1 ring-gray-200',
                      disabled ? 'opacity-60 cursor-not-allowed' : 'text-gray-700 hover:bg-gray-50',
                    ].join(' ')}
                    title={label}
                  >
                    {loading ? (
                      <div className="flex items-center gap-1">
                        <span className="inline-block w-1.5 h-1.5 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                        <span className="inline-block w-1.5 h-1.5 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                        <span className="inline-block w-1.5 h-1.5 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                      </div>
                    ) : (
                      <span className="inline-block h-2 w-2 rounded-full bg-primary-600" />
                    )}
                    {label}
                  </button>
                );
              })}
            </div>

            {activeChipLoading && (
              <div className="mt-4 flex items-center justify-center gap-2">
                <div className="text-sm text-gray-600">Generating: <span className="font-medium">{activeChipLoading}</span></div>
                <div className="flex gap-1">
                  <span className="inline-block w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="inline-block w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="inline-block w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            )}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Generate Summary</h1>
        <p className="text-gray-600">Upload or paste text, then select from suggestions. Results appear as a chat.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="card animate-slide-up">
            <div className="flex border-b border-gray-200 mb-6">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`tab ${activeTab === tab.id ? 'tab-active' : ''}`}
                  disabled={isLoadingInitial || !!activeChipLoading || ragLoading}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {activeTab === 'text' && (
              <TextInput onSubmit={handleSubmit} isLoading={isLoadingInitial || !!activeChipLoading || ragLoading} />
            )}

            {activeTab === 'file' && (
              <FileUpload
                onSubmit={handleSubmit}
                isLoading={isLoadingInitial || !!activeChipLoading || ragLoading}
                acceptedTypes={['.pdf', '.doc', '.docx']}
                fileType="text"
                title="Upload Document"
                maxFileSize="50 MB"
              />
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="card animate-slide-up">
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="text-xl font-semibold text-gray-800">Summary Output</h2>

              <button
                type="button"
                onClick={handleDownloadHistory}
                disabled={!history || history.length === 0}
                className={[
                  'inline-flex items-center gap-2 rounded-lg bg-primary-600 px-3 py-2 text-sm font-medium text-white shadow-sm',
                  !history || history.length === 0 ? 'opacity-60 cursor-not-allowed' : 'hover:bg-primary-700',
                ].join(' ')}
                title="Download generated output as PDF"
              >
                <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path d="M10 12a1 1 0 0 1-1-1V3a1 1 0 1 1 2 0v8a1 1 0 0 1-1 1Z" />
                  <path d="M6.707 9.293a1 1 0 0 1 1.414 0L10 11.172l1.879-1.879a1 1 0 1 1 1.414 1.414l-2.586 2.586a1 1 0 0 1-1.414 0L6.707 10.707a1 1 0 0 1 0-1.414Z" />
                  <path d="M4 14a1 1 0 0 1 1 1v1h10v-1a1 1 0 1 1 2 0v2a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-2a1 1 0 0 1 1-1Z" />
                </svg>
                Download PDF
              </button>
            </div>

            <div className="min-h-[400px] max-h-[700px] overflow-y-auto">{renderChat()}</div>

            {/* Chat Input */}
            <div className="mt-4">
              <form onSubmit={handleAskRag} className="flex flex-col gap-3 sm:flex-row">
                <input
                  value={ragQuestion}
                  onChange={(e) => setRagQuestion(e.target.value)}
                  onFocus={() => setChatFocused(true)}
                  onBlur={() => setChatFocused(false)}
                  placeholder="Ask a question..."
                  disabled={!docId || ragLoading || isLoadingInitial || !!activeChipLoading}
                  className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-primary-300"
                />
                <button
                  type="submit"
                  disabled={!docId || ragLoading || isLoadingInitial || !!activeChipLoading}
                  className={[
                    'inline-flex items-center justify-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white shadow-sm',
                    !docId || ragLoading || isLoadingInitial || !!activeChipLoading
                      ? 'opacity-60 cursor-not-allowed'
                      : 'hover:bg-primary-700',
                  ].join(' ')}
                >
                  {ragLoading ? (
                    <div className="flex items-center gap-1">
                      <span className="inline-block w-1.5 h-1.5 bg-white rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                      <span className="inline-block w-1.5 h-1.5 bg-white rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                      <span className="inline-block w-1.5 h-1.5 bg-white rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                    </div>
                  ) : null}
                  Ask
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Generate;
