import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
};

// RNF-74/75 — @next/bundle-analyzer é um plugin de WEBPACK: injeta uma
// função `webpack()` na config. Next.js 16 usa Turbopack por padrão em
// `next build`, e Turbopack NÃO reconhece `webpack()` (a doc oficial de
// upgrade avisa que isso pode até falhar o build "para prevenir
// misconfiguration"). Por isso o wrapper só é aplicado quando ANALYZE=true
// é setado explicitamente — o `pnpm build` padrão (Turbopack, usado em
// produção) nunca vê essa config; só uma invocação diagnóstica dedicada
// (`ANALYZE=true pnpm build --webpack`, ver README/RELATORIO-RNF-74-RNF-75.md)
// usa o analyzer, com `--webpack` forçando o único bundler que o plugin
// suporta.
export default process.env.ANALYZE === "true"
  ? // eslint-disable-next-line @typescript-eslint/no-require-imports
    (require("@next/bundle-analyzer")({ enabled: true })(
      nextConfig,
    ) as NextConfig)
  : nextConfig;
