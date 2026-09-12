import { asc, desc, eq } from "drizzle-orm";
import { getChatGPTUser } from "../../chatgpt-auth";
import { getDb } from "../../../db";
import { playerSideAssignments } from "../../../db/schema";

function serialize(row: typeof playerSideAssignments.$inferSelect) {
  return {
    id: row.id,
    video_id: row.videoId,
    version: row.version,
    effective_frame: row.effectiveFrame,
    effective_time: row.effectiveTimeMs / 1000,
    upper_player: row.upperPlayer,
    lower_player: row.lowerPlayer,
    source: row.source,
    editor_id: row.editorId,
    created_at: row.createdAt,
  };
}

function routeError(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected player-side error";
  const status = message.includes("no such table") ? 503 : 500;
  return Response.json({ error: status === 503 ? "球员场侧数据库正在初始化，请稍后重试" : message }, { status });
}

export async function GET(request: Request) {
  const videoId = new URL(request.url).searchParams.get("video_id")?.trim();
  if (!videoId) return Response.json({ error: "video_id is required" }, { status: 400 });
  try {
    const rows = await getDb().select().from(playerSideAssignments)
      .where(eq(playerSideAssignments.videoId, videoId))
      .orderBy(asc(playerSideAssignments.effectiveFrame), asc(playerSideAssignments.version));
    return Response.json({ assignments: rows.map(serialize) });
  } catch (error) {
    return routeError(error);
  }
}

export async function POST(request: Request) {
  try {
    const payload = await request.json() as {
      video_id?: string;
      effective_frame?: number;
      effective_time?: number;
      upper_player?: string;
      lower_player?: string;
    };
    const videoId = payload.video_id?.trim();
    const effectiveFrame = Math.round(Number(payload.effective_frame));
    const effectiveTime = Number(payload.effective_time);
    const upperPlayer = payload.upper_player?.trim();
    const lowerPlayer = payload.lower_player?.trim();
    if (!videoId) return Response.json({ error: "video_id is required" }, { status: 400 });
    if (!Number.isInteger(effectiveFrame) || effectiveFrame < 0 || !Number.isFinite(effectiveTime) || effectiveTime < 0) {
      return Response.json({ error: "effective frame and time must be non-negative" }, { status: 400 });
    }
    if (!upperPlayer || !lowerPlayer || !["A", "B"].includes(upperPlayer) || !["A", "B"].includes(lowerPlayer) || upperPlayer === lowerPlayer) {
      return Response.json({ error: "upper_player and lower_player must be different A/B values" }, { status: 400 });
    }
    const db = getDb();
    const [latest] = await db.select({ version: playerSideAssignments.version }).from(playerSideAssignments)
      .where(eq(playerSideAssignments.videoId, videoId)).orderBy(desc(playerSideAssignments.version)).limit(1);
    const user = await getChatGPTUser();
    const record: typeof playerSideAssignments.$inferInsert = {
      id: `side_${crypto.randomUUID()}`,
      videoId,
      version: (latest?.version ?? 0) + 1,
      effectiveFrame,
      effectiveTimeMs: Math.round(effectiveTime * 1000),
      upperPlayer,
      lowerPlayer,
      source: "manual_side_switch",
      editorId: user?.userId ?? "public-annotator",
      createdAt: new Date().toISOString(),
    };
    const [saved] = await db.insert(playerSideAssignments).values(record).returning();
    return Response.json({ assignment: serialize(saved) }, { status: 201 });
  } catch (error) {
    return routeError(error);
  }
}
