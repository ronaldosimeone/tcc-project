import LoginForm from "@/components/auth/login-form";

/** Login (RBAC). Server Component: só resolve `?next=` e delega ao formulário. */
export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  const { next } = await searchParams;
  const target =
    typeof next === "string" && next.startsWith("/") && !next.startsWith("//")
      ? next
      : "/";
  return (
    <main className="flex min-h-dvh items-center justify-center bg-background px-4 text-foreground">
      <LoginForm next={target} />
    </main>
  );
}
