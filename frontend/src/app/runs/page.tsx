import type { Metadata } from "next";

import RunsBrowser from "@/components/RunsBrowser";

export const metadata: Metadata = {
  title: "Runs · Global-MMLU multilingual evaluation",
};

export default function RunsPage() {
  return <RunsBrowser />;
}
