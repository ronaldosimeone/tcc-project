import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { MswProvider } from "@/components/msw-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "PredictIQ — Sistema de Manutenção Preditiva",
  description:
    "Monitoramento em tempo real e predição de falhas para compressores industriais MetroPT-3. TCC · Indústria 4.0.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="pt-BR"
      className={`${geistSans.variable} ${geistMono.variable} dark h-full`}
    >
      <body className="h-full antialiased">
        <MswProvider>{children}</MswProvider>
      </body>
    </html>
  );
}
