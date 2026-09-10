// ── Markdown seguro — RNF-58: extraído de maintenance-assistant.tsx ─────────
//
// react-markdown, por padrão, NUNCA interpreta HTML bruto do texto de origem
// (nenhum `rehype-raw` é usado aqui) — um `<script>...</script>` no Markdown
// vira texto literal, não um elemento executado. O único ponto de risco real
// é o `href` de um link Markdown (`[texto](url)`) virar um `<a>` de verdade.
// `isSafeHref` bloqueia `javascript:`, `data:` e qualquer esquema que não
// seja http/https antes de renderizar como link clicável.

import type { Components } from "react-markdown";

export function isSafeHref(href: string): boolean {
  try {
    const url = new URL(href, "http://localhost");
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

// Sem plugin de tipografia (`@tailwindcss/typography` não é usado em
// nenhum outro lugar do projeto) — classes explícitas por elemento,
// reaproveitando só os tokens de cor/tema já existentes (`text-foreground`,
// `text-muted-foreground`), em vez de puxar uma dependência nova só para
// este painel.
export const markdownComponents: Components = {
  h1: ({ children, ...props }) => (
    <h1 className="mb-2 text-lg font-bold text-foreground" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2
      className="mt-4 mb-1.5 text-base font-semibold text-foreground"
      {...props}
    >
      {children}
    </h2>
  ),
  h3: ({ children, ...props }) => (
    <h3 className="mt-3 mb-1 text-sm font-semibold text-foreground" {...props}>
      {children}
    </h3>
  ),
  p: ({ children, ...props }) => (
    <p className="mb-2 text-sm leading-relaxed text-foreground" {...props}>
      {children}
    </p>
  ),
  ul: ({ children, ...props }) => (
    <ul
      className="mb-2 list-disc space-y-1 pl-5 text-sm text-foreground"
      {...props}
    >
      {children}
    </ul>
  ),
  ol: ({ children, ...props }) => (
    <ol
      className="mb-2 list-decimal space-y-1 pl-5 text-sm text-foreground"
      {...props}
    >
      {children}
    </ol>
  ),
  code: ({ children, ...props }) => (
    <code
      className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs"
      {...props}
    >
      {children}
    </code>
  ),
  a: ({ href, children, ...props }) => {
    if (!href || !isSafeHref(href)) {
      // Link inseguro/sem URL — nunca renderiza como <a>, mostra só o texto.
      return <span {...props}>{children}</span>;
    }
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-primary underline underline-offset-2"
        {...props}
      >
        {children}
      </a>
    );
  },
};
