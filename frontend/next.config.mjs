/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required for Docker multi-stage build (Dockerfile uses .next/standalone)
  output: "standalone",

  // Allow Steam CDN images to load in <img> tags
  // (We use native <img> not next/image to avoid domain config complexity in Phase 1)
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "cdn.cloudflare.steamstatic.com",
      },
      {
        protocol: "https",
        hostname: "cdn.akamai.steamstatic.com",
      },
    ],
  },
};

export default nextConfig;
