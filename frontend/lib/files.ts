// Read a browser File into a data URL (base64) so it can be embedded as an
// OpenAI-compatible image_url part, and uploaded to /v1/files when available.

import type { Attachment } from "./types";

export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

let _seq = 0;
function id(): string {
  _seq += 1;
  return `att_${Date.now().toString(36)}_${_seq}`;
}

export async function toAttachment(file: File): Promise<Attachment> {
  const dataUrl = await readFileAsDataUrl(file);
  return {
    id: id(),
    name: file.name,
    mime: file.type || "application/octet-stream",
    size: file.size,
    dataUrl,
    isImage: (file.type || "").startsWith("image/"),
  };
}
