import { cn } from "@/lib/utils";

/**
 * Skeleton — componente de loading state (RNF-21).
 *
 * Design translúcido estilo Vercel/Linear: `bg-foreground/8` cria um véu
 * neutro que herda o contraste do tema sem cores hard-coded. Em light mode
 * foreground é quase-preto → o tint fica cinza-claro sutil. Em dark mode
 * foreground é quase-branco → o tint fica cinza-escuro sutil.
 * Resultado: visível em ambos os temas, sem chocar com nenhuma palette.
 */
function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-foreground/8", className)}
      aria-hidden="true"
      {...props}
    />
  );
}

export { Skeleton };
