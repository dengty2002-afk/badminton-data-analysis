import vinext from "vinext";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import hostingConfig from "./hosting.local.json";
import { sites } from "./build/sites-vite-plugin";

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  "00000000-0000-4000-8000-000000000000";

const { d1, r2 } = hostingConfig;

function localAnnotationVideoPlugin(): Plugin {
  return {
    name: "shuttlelab-local-annotation-videos",
    apply: "serve",
    configureServer(server) {
      const catalogDir = resolve(process.cwd(), "..", "data", "videos");
      const goldSnapshotDir = resolve(process.cwd(), "..", "data", "gold", "play-state-prospective-001");

      server.middlewares.use("/__local_gold_bundle", (request, response) => {
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end("Method not allowed");
          return;
        }
        let body = "";
        request.setEncoding("utf8");
        request.on("data", (chunk: string) => {
          body += chunk;
          if (body.length > 5_000_000) request.destroy();
        });
        request.on("end", () => {
          try {
            const payload = JSON.parse(body) as { schemaVersion?: string; batchId?: string };
            if (payload.schemaVersion !== "shuttlelab-gold-bundle-1" || payload.batchId !== "play-state-prospective-001") {
              response.statusCode = 400;
              response.end("Invalid Gold bundle");
              return;
            }
            mkdirSync(goldSnapshotDir, { recursive: true });
            const snapshotName = `browser-export-${Date.now()}.json`;
            writeFileSync(resolve(goldSnapshotDir, snapshotName), `${JSON.stringify(payload, null, 2)}\n`, "utf8");
            response.setHeader("Content-Type", "application/json");
            response.end(JSON.stringify({ ok: true, snapshotName }));
          } catch {
            response.statusCode = 400;
            response.end("Invalid JSON");
          }
        });
      });

      server.middlewares.use("/__local_video", (request, response) => {
        const videoId = (request.url ?? "").split("?")[0].replace(/^\//, "");
        if (!/^vid_[a-z0-9]+$/.test(videoId)) {
          response.statusCode = 400;
          response.end("Invalid video id");
          return;
        }

        try {
          const catalogPath = resolve(catalogDir, `${videoId}.json`);
          const metadata = JSON.parse(readFileSync(catalogPath, "utf8")) as { path?: string };
          if (!metadata.path || !existsSync(metadata.path)) {
            response.statusCode = 404;
            response.end("Registered local video is unavailable");
            return;
          }

          const size = statSync(metadata.path).size;
          const range = request.headers.range;
          response.setHeader("Content-Type", "video/mp4");
          response.setHeader("Accept-Ranges", "bytes");
          response.setHeader("Cache-Control", "private, max-age=0, must-revalidate");

          if (range) {
            const match = /^bytes=(\d*)-(\d*)$/.exec(range);
            if (!match) {
              response.statusCode = 416;
              response.setHeader("Content-Range", `bytes */${size}`);
              response.end();
              return;
            }
            const start = match[1] ? Number(match[1]) : 0;
            const end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
            if (start > end || start >= size) {
              response.statusCode = 416;
              response.setHeader("Content-Range", `bytes */${size}`);
              response.end();
              return;
            }
            response.statusCode = 206;
            response.setHeader("Content-Length", end - start + 1);
            response.setHeader("Content-Range", `bytes ${start}-${end}/${size}`);
            if (request.method === "HEAD") response.end();
            else createReadStream(metadata.path, { start, end }).pipe(response);
            return;
          }

          response.statusCode = 200;
          response.setHeader("Content-Length", size);
          if (request.method === "HEAD") response.end();
          else createReadStream(metadata.path).pipe(response);
        } catch {
          response.statusCode = 404;
          response.end("Video is not registered");
        }
      });
    },
  };
}

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";

const localBindingConfig = {
  main: "./worker/index.ts",
  compatibility_flags: ["nodejs_compat"],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: "site-creator-d1",
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: "site-creator-r2",
        },
      ]
    : [],
};

export default defineConfig(async () => {
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= "false";
  process.env.WRANGLER_LOG_PATH ??= ".wrangler/logs";
  process.env.MINIFLARE_REGISTRY_PATH ??= ".wrangler/registry";

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import("@cloudflare/vite-plugin");

  return {
    server: {
      host: "0.0.0.0",
      watch: isCodexSeatbeltSandbox
        ? { useFsEvents: false, usePolling: true, ignored: ["**/public/example.mp4"] }
        : { ignored: ["**/public/example.mp4"] },
    },
    plugins: [
      localAnnotationVideoPlugin(),
      vinext(),
      sites(),
      cloudflare({
        viteEnvironment: { name: "rsc", childEnvironments: ["ssr"] },
        config: localBindingConfig,
      }),
    ],
  };
});
