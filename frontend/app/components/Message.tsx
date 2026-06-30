"use client";

import { useState } from "react";
import type { ChatMessage } from "@/lib/types";

function IconButton({
  title,
  active,
  onClick,
  children,
}: {
  title: string;
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={`rounded-md px-2 py-1 text-xs transition-colors hover:bg-surface-2 ${
        active ? "text-accent" : "text-muted"
      }`}
    >
      {children}
    </button>
  );
}

export function Message({
  message,
  onFeedback,
  onEdit,
  onRegenerate,
}: {
  message: ChatMessage;
  onFeedback: (id: string, rating: "up" | "down") => void;
  onEdit: (id: string, text: string) => void;
  onRegenerate: (id: string) => void;
}) {
  const isUser = message.role === "user";
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm ${
            isUser
              ? "rounded-br-sm text-accent-fg"
              : "rounded-bl-sm border border-border bg-surface"
          }`}
          style={isUser ? { background: "var(--accent)" } : undefined}
        >
          {/* image attachments on user messages */}
          {isUser && message.attachments && message.attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {message.attachments.map((a) =>
                a.isImage ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    key={a.id}
                    src={a.dataUrl}
                    alt={a.name}
                    className="max-h-40 rounded-lg border border-white/20"
                  />
                ) : (
                  <span
                    key={a.id}
                    className="rounded-lg bg-black/20 px-2 py-1 text-xs"
                  >
                    📎 {a.name}
                  </span>
                )
              )}
            </div>
          )}

          {editing ? (
            <div className="flex flex-col gap-2">
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                rows={Math.min(10, Math.max(2, draft.split("\n").length))}
                className="w-72 rounded-lg border border-border bg-surface-2 p-2 text-fg outline-none focus:border-accent"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    onEdit(message.id, draft);
                    setEditing(false);
                  }}
                  className="rounded-md px-3 py-1 text-xs font-medium text-accent-fg"
                  style={{ background: "var(--accent)" }}
                >
                  Save &amp; submit
                </button>
                <button
                  onClick={() => {
                    setDraft(message.content);
                    setEditing(false);
                  }}
                  className="rounded-md border border-border px-3 py-1 text-xs"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className={`msg-body ${message.pending ? "caret" : ""}`}>
              {message.content || (message.pending ? "" : "")}
            </div>
          )}
        </div>

        {/* action bar */}
        {!editing && (
          <div
            className={`mt-1 flex items-center gap-1 ${
              isUser ? "justify-end" : "justify-start"
            }`}
          >
            {isUser ? (
              <IconButton title="Edit & resubmit" onClick={() => setEditing(true)}>
                ✎ edit
              </IconButton>
            ) : (
              !message.pending && (
                <>
                  <IconButton
                    title="Good response"
                    active={message.feedback === "up"}
                    onClick={() => onFeedback(message.id, "up")}
                  >
                    👍
                  </IconButton>
                  <IconButton
                    title="Bad response"
                    active={message.feedback === "down"}
                    onClick={() => onFeedback(message.id, "down")}
                  >
                    👎
                  </IconButton>
                  <IconButton
                    title="Regenerate"
                    onClick={() => onRegenerate(message.id)}
                  >
                    ↻ regenerate
                  </IconButton>
                  {message.model && (
                    <span className="px-1 text-[11px] text-muted">
                      {message.model}
                    </span>
                  )}
                </>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}
