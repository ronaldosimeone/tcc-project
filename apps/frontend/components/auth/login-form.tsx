"use client";

/**
 * Formulário de login — troca usuário/senha por um JWT (`POST /v1/auth/login`)
 * e volta para `next`. Mensagem de erro única para usuário inexistente e
 * senha errada (o backend também não diferencia).
 */

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Loader2, LogIn } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { login } from "@/lib/auth";

interface LoginFormProps {
  next: string;
}

export default function LoginForm({ next }: LoginFormProps) {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setPending(true);
    setError(null);
    try {
      await login({ username, password });
      router.replace(next);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Falha no login.");
      setPending(false);
    }
  };

  return (
    <Card className="w-full max-w-sm">
      <CardContent className="p-6">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          PredictIQ
        </p>
        <h1 className="mt-1 text-xl font-bold tracking-tight text-foreground">
          Entrar
        </h1>
        <form className="mt-5 space-y-4" onSubmit={(e) => void onSubmit(e)}>
          <div className="space-y-1.5">
            <Label htmlFor="login-username">Usuário</Label>
            <Input
              id="login-username"
              autoComplete="username"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="login-password">Senha</Label>
            <Input
              id="login-password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          {error && (
            <p role="alert" className="text-xs text-destructive">
              {error}
            </p>
          )}
          <Button type="submit" className="w-full" disabled={pending}>
            {pending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <LogIn className="h-4 w-4" />
            )}
            Entrar
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
