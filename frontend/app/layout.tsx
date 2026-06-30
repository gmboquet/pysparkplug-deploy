import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";
import { AuthProvider } from "@/lib/auth";

export const metadata: Metadata = {
  title: "mixle — calibrated AI platform",
  description:
    "Host mixle probabilistic models + open LLMs behind one gateway. Calibrated, with a real feedback loop.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-fg antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
