import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const base = `${protocol}://${host}`;

  return {
    title: "ShuttleLab · 羽毛球逐拍数据工作台",
    description: "基于人工击球锚点，盲标18类球种与米制目的地的逐拍语义 Gold 工作台。",
    icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
    openGraph: {
      title: "ShuttleLab · 羽毛球逐拍数据工作台",
      description: "盲标18类球种、下一次触球与终局落点，生成可追溯逐拍语义 Gold。",
      type: "website",
      images: [{ url: `${base}/og-semantic-gold.png`, width: 1536, height: 1024, alt: "ShuttleLab 逐拍语义 Gold 工作台" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "ShuttleLab · 羽毛球逐拍数据工作台",
      description: "盲标18类球种、下一次触球与终局落点，生成可追溯逐拍语义 Gold。",
      images: [`${base}/og-semantic-gold.png`],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
