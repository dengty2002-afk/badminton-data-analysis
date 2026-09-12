import { desc, eq } from "drizzle-orm";
import { getChatGPTUser } from "../../chatgpt-auth";
import { getDb } from "../../../db";
import { calibrations } from "../../../db/schema";
import { computeHomography, validateCalibration, type CalibrationPoint } from "../../../lib/homography";

function serialize(row: typeof calibrations.$inferSelect) {
  return {
    id: row.id,
    video_id: row.videoId,
    version: row.version,
    frame_time_ms: row.frameTimeMs,
    image_width: row.imageWidth,
    image_height: row.imageHeight,
    corners: JSON.parse(row.cornersJson) as CalibrationPoint[],
    homography: JSON.parse(row.homographyJson) as number[][],
    court_orientation: row.courtOrientation,
    quality_status: row.qualityStatus,
    source: row.source,
    calibrator_id: row.calibratorId,
    created_at: row.createdAt,
  };
}

function routeError(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected calibration error";
  const status = message.includes("no such table") ? 503 : 500;
  return Response.json({ error: status === 503 ? "标定数据库正在初始化，请稍后重试" : message }, { status });
}

export async function GET(request: Request) {
  const videoId = new URL(request.url).searchParams.get("video_id")?.trim();
  if (!videoId) return Response.json({ error: "video_id is required" }, { status: 400 });
  try {
    const rows = await getDb().select().from(calibrations).where(eq(calibrations.videoId, videoId)).orderBy(desc(calibrations.version));
    return Response.json({ calibrations: rows.map(serialize) });
  } catch (error) {
    return routeError(error);
  }
}

export async function POST(request: Request) {
  try {
    const payload = await request.json() as {
      video_id?: string;
      frame_time_ms?: number;
      image_width?: number;
      image_height?: number;
      corners?: CalibrationPoint[];
    };
    const videoId = payload.video_id?.trim();
    const width = Number(payload.image_width);
    const height = Number(payload.image_height);
    const corners = payload.corners ?? [];
    if (!videoId) return Response.json({ error: "video_id is required" }, { status: 400 });
    const validation = validateCalibration(corners, width, height);
    if (!validation.valid) return Response.json({ error: validation.message }, { status: 400 });

    const db = getDb();
    const [latest] = await db.select({ version: calibrations.version }).from(calibrations).where(eq(calibrations.videoId, videoId)).orderBy(desc(calibrations.version)).limit(1);
    const user = await getChatGPTUser();
    const record: typeof calibrations.$inferInsert = {
      id: `cal_${crypto.randomUUID()}`,
      videoId,
      version: (latest?.version ?? 0) + 1,
      frameTimeMs: Math.max(0, Math.round(Number(payload.frame_time_ms) || 0)),
      imageWidth: Math.round(width),
      imageHeight: Math.round(height),
      cornersJson: JSON.stringify(corners),
      homographyJson: JSON.stringify(computeHomography(corners)),
      courtOrientation: "upper_is_far_side",
      qualityStatus: "accepted",
      source: "manual",
      calibratorId: user?.userId ?? "local-private-user",
      createdAt: new Date().toISOString(),
    };
    const [saved] = await db.insert(calibrations).values(record).returning();
    return Response.json({ calibration: serialize(saved) }, { status: 201 });
  } catch (error) {
    return routeError(error);
  }
}
