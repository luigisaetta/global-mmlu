import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server allows only localhost by default; also accept the UI
  // opened as http://127.0.0.1:3000.
  allowedDevOrigins: ["127.0.0.1"],
  cacheComponents: true,
  partialPrefetching: true,
};

export default nextConfig;
