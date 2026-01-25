// src/services/api.js
const BACKEND_ENDPOINT = import.meta.env.VITE_BACKEND_ENDPOINT || '';

// Helper function to decode Python string escape sequences
const decodePythonString = (str) => {
  return str
    .replace(/\\x([0-9a-fA-F]{2})/g, (match, hex) => String.fromCharCode(parseInt(hex, 16)))
    .replace(/\\n/g, '\n')
    .replace(/\\r/g, '\r')
    .replace(/\\t/g, '\t')
    .replace(/\\"/g, '"')
    .replace(/\\'/g, "'")
    .replace(/\\\\/g, '\\');
};

// Clone FormData safely (used by UI)
export const cloneFormData = (formData) => {
  const fd = new FormData();
  for (const [k, v] of formData.entries()) fd.append(k, v);
  return fd;
};

/**
 * Helper to clone FormData and add mode/section/doc_id
 */
export const withModeAndSection = (formData, mode, section, docId) => {
  const fd = cloneFormData(formData);
  if (mode) fd.set('mode', mode);
  if (section) fd.set('section', section);
  if (docId) fd.set('doc_id', docId);
  return fd;
};

/**
 * Parse SSE-like response content into a JSON object if possible.
 */
const parseSseLikeToJson = (responseText) => {
  const lines = responseText.split('\n');
  let extractedText = '';
  let extractedDocId = '';
  let extractedSections = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.includes('[DONE]') || trimmed === '' || trimmed === 'data:') continue;

    if (trimmed.startsWith('data: ')) {
      let dataContent = trimmed.substring(6).trim();

      if (dataContent.startsWith("b'") && dataContent.endsWith("'")) {
        dataContent = dataContent.substring(2, dataContent.length - 1);
      }
      if (dataContent.startsWith('b"') && dataContent.endsWith('"')) {
        dataContent = dataContent.substring(2, dataContent.length - 1);
      }

      const tryParse = (s) => {
        try {
          return JSON.parse(s);
        } catch {
          return null;
        }
      };

      let obj = tryParse(dataContent);

      if (!obj) {
        try {
          const decoded = decodePythonString(dataContent);
          obj = tryParse(decoded);
        } catch {
          obj = null;
        }
      }

      if (obj) {
        if (obj.text) extractedText += obj.text;
        if (obj.doc_id) extractedDocId = obj.doc_id;
        if (obj.sections || obj.section_options) extractedSections = obj.sections || obj.section_options;
      } else {
        // best-effort text fallback
        const textMatch = dataContent.match(/"text":"([^"]*(?:\\.[^"]*)*)"/);
        if (textMatch && textMatch[1]) {
          const unescapedText = textMatch[1]
            .replace(/\\"/g, '"')
            .replace(/\\\\/g, '\\')
            .replace(/\\n/g, '\n');
          extractedText += unescapedText;
        }
      }
    }
  }

  return {
    text: extractedText || responseText,
    doc_id: extractedDocId || '',
    sections: extractedSections || undefined,
  };
};

/**
 * Generate and return JSON
 * Backend should ideally return:
 * { text: "...", doc_id: "...", sections: ["A","B","C"] }
 */
export const generateSummaryJson = async (formData) => {
  const response = await fetch(`${BACKEND_ENDPOINT}/v1/docsum`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} - ${errorText}`);
  }

  const responseText = await response.text();

  // Handle SSE-like response
  if (responseText.includes('data:')) {
    return parseSseLikeToJson(responseText);
  }

  // Normal JSON response
  try {
    return JSON.parse(responseText);
  } catch {
    // Backend returned non-JSON text
    return { text: responseText, doc_id: '' };
  }
};

/**
 * Backwards compatible: return only text (existing callers)
 */
export const generateSummary = async (formData) => {
  const json = await generateSummaryJson(formData);
  return json.text || json.output_text || '';
};

export const generateSummaryStreaming = async (formData, onChunk) => {
  const response = await fetch(`${BACKEND_ENDPOINT}/v1/docsum`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} - ${errorText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.includes('[DONE]') || trimmed === '' || trimmed === 'data:') continue;

      if (trimmed.startsWith('data: ')) {
        let dataContent = trimmed.substring(6).trim();

        if (dataContent.startsWith("b'") && dataContent.endsWith("'")) {
          dataContent = dataContent.substring(2, dataContent.length - 1);
        }
        if (dataContent.startsWith('b"') && dataContent.endsWith('"')) {
          dataContent = dataContent.substring(2, dataContent.length - 1);
        }

        try {
          const data = JSON.parse(dataContent);
          if (data.text) onChunk(data.text);
        } catch {
          if (dataContent) onChunk(decodePythonString(dataContent));
        }
      }
    }
  }
};

/**
 * Convenience wrappers
 */
export const generateInitialSummary = async (formData) => {
  return generateSummaryJson(withModeAndSection(formData, 'financial_initial'));
};

export const generateSectionSummary = async (formData, section) => {
  return generateSummaryJson(withModeAndSection(formData, 'financial_section', section));
};

/**
 * RAG status: GET /v1/rag/status?doc_id=...
 */
export const getRagStatus = async (docId) => {
  const id = (docId || '').trim();
  if (!id) return { ready: false };

  const response = await fetch(`${BACKEND_ENDPOINT}/v1/rag/status?doc_id=${encodeURIComponent(id)}`);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`RAG Status Error: ${response.status} - ${errorText}`);
  }

  return response.json();
};

/**
 * RAG chat: POST /v1/rag/chat (doc_id + message)
 */
export const ragChat = async ({ docId, message, maxTokens = 500, temperature = 0.2 }) => {
  const fd = new FormData();
  fd.set('doc_id', (docId || '').trim());
  fd.set('message', (message || '').trim());
  fd.set('max_tokens', String(maxTokens));
  fd.set('temperature', String(temperature));

  const response = await fetch(`${BACKEND_ENDPOINT}/v1/rag/chat`, {
    method: 'POST',
    body: fd,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`RAG Chat Error: ${response.status} - ${errorText}`);
  }

  return response.json();
};
/**
 * Delete vectors: DELETE /v1/vectors/{doc_id}
 * Clean up old vector embeddings before uploading a new document
 */
export const deleteVectors = async (docId) => {
  const id = (docId || '').trim();
  if (!id) return { status: 'skipped' };

  try {
    const response = await fetch(`${BACKEND_ENDPOINT}/v1/vectors/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      console.warn(`Failed to delete vectors for ${id}: ${response.status}`);
      return { status: 'failed' };
    }

    return response.json();
  } catch (e) {
    console.warn(`Vector cleanup error: ${e.message}`);
    return { status: 'error', error: e.message };
  }
};