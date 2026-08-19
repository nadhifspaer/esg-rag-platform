/** @type {import('next').NextConfig} */
const nextConfig = {
  // Site root now lands on chat; `/about` (the info page) is reachable from the sidebar instead.
  async redirects() {
    return [
      {
        source: "/",
        destination: "/chat",
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
