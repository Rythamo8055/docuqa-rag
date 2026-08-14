/**
 * Client-side upload validation — mirrors the backend rules so users get
 * instant feedback before the file travels anywhere. The backend still
 * validates independently (never trust the client).
 */

export type UploadError =
  | "NO_FILE"
  | "WRONG_TYPE"
  | "TOO_LARGE"
  | "BAD_MAGIC"
  | "EMPTY"
  | "READ_FAILED";

export interface UploadCheck {
  ok: boolean;
  error?: UploadError;
  message?: string;
}

const MAX_SIZE_MB = 20;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

export async function validatePdf(file: File): Promise<UploadCheck> {
  if (!file) {
    return { ok: false, error: "NO_FILE", message: "No file selected." };
  }
  if (file.size === 0) {
    return { ok: false, error: "EMPTY", message: "The file is empty." };
  }
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    return {
      ok: false,
      error: "WRONG_TYPE",
      message: "Only PDF files are allowed. You selected “" + file.name + "”.",
    };
  }
  if (file.size > MAX_SIZE_BYTES) {
    return {
      ok: false,
      error: "TOO_LARGE",
      message: `File is ${(file.size / 1024 / 1024).toFixed(1)} MB — the limit is ${MAX_SIZE_MB} MB.`,
    };
  }

  // Magic bytes: PDFs start with %PDF (allow leading whitespace)
  try {
    const head = new Uint8Array(await file.slice(0, 1024).arrayBuffer());
    let i = 0;
    while (i < head.length && (head[i] === 0x20 || head[i] === 0x09 || head[i] === 0x0a || head[i] === 0x0d)) {
      i++;
    }
    const magic = String.fromCharCode(head[i], head[i + 1], head[i + 2], head[i + 3]);
    if (magic !== "%PDF") {
      return {
        ok: false,
        error: "BAD_MAGIC",
        message: "This file is not a valid PDF (its content does not start with %PDF).",
      };
    }
  } catch {
    return {
      ok: false,
      error: "READ_FAILED",
      message: "Could not read the file. Please try again.",
    };
  }

  return { ok: true };
}

export const UPLOAD_ERROR_COPY: Record<UploadError, { title: string; hint: string }> = {
  NO_FILE: { title: "No file selected", hint: "Choose a PDF to begin." },
  WRONG_TYPE: { title: "Not a PDF", hint: "Export the document as a PDF and upload it again." },
  TOO_LARGE: { title: "File too large", hint: "The limit is 20 MB. Split the document or compress it." },
  BAD_MAGIC: { title: "Corrupt or fake PDF", hint: "Re-export the document from its source app." },
  EMPTY: { title: "Empty file", hint: "The file contains no data." },
  READ_FAILED: { title: "Could not read file", hint: "Try uploading the file again." },
};