import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Global-MMLU multilingual evaluation",
  description: "Evaluate OCI Generative AI models on Global-MMLU, language by language.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
