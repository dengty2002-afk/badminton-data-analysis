import { and, desc, eq } from "drizzle-orm";
import { env } from "cloudflare:workers";
import { getChatGPTUser } from "../../chatgpt-auth";
import { getDb } from "../../../db";
import { semanticGoldRevisions } from "../../../db/schema";

const STROKE_TYPES = new Set([
  "net shot", "return net", "smash", "wrist smash", "lob", "defensive return lob",
  "clear", "drive", "driven flight", "back-court drive", "drop", "passive drop",
  "push", "rush", "defensive return drive", "cross-court net shot", "short service", "long service",
]);
const HITTERS = new Set(["upper", "lower", "unknown"]);
const LANDING_KINDS = new Set(["next_contact", "terminal"]);
const LANDING_STATUSES = new Set(["observed", "out", "caught", "unobservable"]);
const STATUSES = new Set(["gold", "uncertain"]);
const CONFIDENCES = new Set(["high", "medium", "low"]);

async function ensureSemanticGoldSchema() {
  const d1 = env.DB;
  await d1.batch([
    d1.prepare(`CREATE TABLE IF NOT EXISTS semantic_gold_revisions (
      id text PRIMARY KEY NOT NULL, video_id text NOT NULL, anchor_id text NOT NULL,
      revision integer NOT NULL, rally_id text NOT NULL, hit_frame integer NOT NULL,
      hitter text NOT NULL, stroke_type text DEFAULT '' NOT NULL, landing_x real,
      landing_y real, landing_frame integer, landing_kind text,
      landing_status text DEFAULT 'observed' NOT NULL, landing_area integer,
      landing_area_schema text DEFAULT 'shuttlelab_9x7_v1' NOT NULL,
      confidence text DEFAULT 'high' NOT NULL, uncertainty_frames integer DEFAULT 0 NOT NULL,
      notes text DEFAULT '' NOT NULL, status text NOT NULL,
      annotator_id text DEFAULT 'public-annotator' NOT NULL, created_at text NOT NULL
    )`),
    d1.prepare("CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_gold_video_anchor_revision ON semantic_gold_revisions (video_id, anchor_id, revision)"),
    d1.prepare("CREATE INDEX IF NOT EXISTS idx_semantic_gold_video_status ON semantic_gold_revisions (video_id, status)"),
  ]);
  const columnResult = await d1.prepare("PRAGMA table_info(semantic_gold_revisions)").all<{ name: string }>();
  const columns = new Set(columnResult.results.map((column) => column.name));
  const upgrades = [];
  if (!columns.has("landing_status")) upgrades.push(d1.prepare("ALTER TABLE semantic_gold_revisions ADD COLUMN landing_status text DEFAULT 'observed' NOT NULL"));
  if (!columns.has("landing_area")) upgrades.push(d1.prepare("ALTER TABLE semantic_gold_revisions ADD COLUMN landing_area integer"));
  if (!columns.has("landing_area_schema")) upgrades.push(d1.prepare("ALTER TABLE semantic_gold_revisions ADD COLUMN landing_area_schema text DEFAULT 'shuttlelab_9x7_v1' NOT NULL"));
  if (upgrades.length) await d1.batch(upgrades);
  await d1.prepare("PRAGMA optimize").run();
}

type Payload = {
  video_id?: string; anchor_id?: string; rally_id?: string; hit_frame?: number; hitter?: string;
  stroke_type?: string; landing_kind?: string | null; landing_status?: string;
  landing_area?: number | null; confidence?: string;
  uncertainty_frames?: number; notes?: string; status?: string;
};

function serialize(row: typeof semanticGoldRevisions.$inferSelect) {
  return {
    id: row.id, video_id: row.videoId, anchor_id: row.anchorId, revision: row.revision,
    rally_id: row.rallyId, hit_frame: row.hitFrame, hitter: row.hitter,
    stroke_type: row.strokeType, landing_x: row.landingX, landing_y: row.landingY,
    landing_frame: row.landingFrame, landing_kind: row.landingKind,
    landing_status: row.landingStatus, landing_area: row.landingArea,
    landing_area_schema: row.landingAreaSchema,
    confidence: row.confidence, uncertainty_frames: row.uncertaintyFrames,
    notes: row.notes, status: row.status, annotator_id: row.annotatorId, created_at: row.createdAt,
  };
}

function errorResponse(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected semantic Gold persistence error";
  return Response.json({ error: message.includes("no such table") ? "语义 Gold 数据库正在初始化" : message }, { status: message.includes("no such table") ? 503 : 500 });
}

export async function GET(request: Request) {
  const videoId = new URL(request.url).searchParams.get("video_id")?.trim();
  if (!videoId) return Response.json({ error: "video_id is required" }, { status: 400 });
  try {
    await ensureSemanticGoldSchema();
    const rows = await getDb().select().from(semanticGoldRevisions)
      .where(eq(semanticGoldRevisions.videoId, videoId)).orderBy(desc(semanticGoldRevisions.revision));
    const latest = new Map<string, typeof semanticGoldRevisions.$inferSelect>();
    for (const row of rows) if (!latest.has(row.anchorId)) latest.set(row.anchorId, row);
    return Response.json({ annotations: [...latest.values()].map(serialize) });
  } catch (error) { return errorResponse(error); }
}

export async function POST(request: Request) {
  try {
    const payload = await request.json() as Payload;
    const videoId = payload.video_id?.trim() ?? "";
    const anchorId = payload.anchor_id?.trim() ?? "";
    const rallyId = payload.rally_id?.trim() || "unknown";
    const hitFrame = Math.round(Number(payload.hit_frame));
    const hitter = payload.hitter?.trim() ?? "";
    const strokeType = payload.stroke_type?.trim().toLowerCase() ?? "";
    const landingKind = payload.landing_kind?.trim() ?? null;
    const landingStatus = payload.landing_status?.trim() ?? "";
    const landingArea = payload.landing_area == null ? null : Math.round(Number(payload.landing_area));
    const status = payload.status?.trim() ?? "";
    const confidence = payload.confidence?.trim() ?? "high";
    const uncertainty = Math.round(Number(payload.uncertainty_frames ?? 0));
    if (!videoId || !anchorId || !Number.isInteger(hitFrame) || hitFrame < 0) return Response.json({ error: "invalid anchor" }, { status: 400 });
    if (!HITTERS.has(hitter) || !STATUSES.has(status) || !CONFIDENCES.has(confidence)) return Response.json({ error: "invalid categorical field" }, { status: 400 });
    if (!Number.isInteger(uncertainty) || uncertainty < 0 || uncertainty > 30) return Response.json({ error: "invalid uncertainty" }, { status: 400 });
    if (status === "gold") {
      if (!STROKE_TYPES.has(strokeType)) return Response.json({ error: "请选择官方18类球种" }, { status: 400 });
      if (!landingKind || !LANDING_KINDS.has(landingKind)) return Response.json({ error: "请选择目的地类型" }, { status: 400 });
      if (!LANDING_STATUSES.has(landingStatus)) return Response.json({ error: "请选择目的地状态" }, { status: 400 });
      if (landingStatus === "observed" && (!Number.isInteger(landingArea) || landingArea! < 1 || landingArea! > 9)) return Response.json({ error: "请选择场内九宫格区域" }, { status: 400 });
      if (landingStatus !== "observed" && landingArea != null) return Response.json({ error: "出界、被接住或看不清时不能填写区域" }, { status: 400 });
    }
    await ensureSemanticGoldSchema();
    const db = getDb();
    const [latest] = await db.select({ revision: semanticGoldRevisions.revision })
      .from(semanticGoldRevisions)
      .where(and(eq(semanticGoldRevisions.videoId, videoId), eq(semanticGoldRevisions.anchorId, anchorId)))
      .orderBy(desc(semanticGoldRevisions.revision)).limit(1);
    const user = await getChatGPTUser();
    const record: typeof semanticGoldRevisions.$inferInsert = {
      id: `sem_${crypto.randomUUID()}`, videoId, anchorId, revision: (latest?.revision ?? 0) + 1,
      rallyId, hitFrame, hitter, strokeType, landingX: null, landingY: null, landingFrame: null,
      landingKind, landingStatus, landingArea, landingAreaSchema: "shuttlelab_9_v2",
      confidence, uncertaintyFrames: uncertainty, notes: payload.notes?.trim() || "",
      status, annotatorId: user?.userId ?? "public-annotator", createdAt: new Date().toISOString(),
    };
    const [saved] = await db.insert(semanticGoldRevisions).values(record).returning();
    return Response.json({ annotation: serialize(saved) }, { status: 201 });
  } catch (error) { return errorResponse(error); }
}
