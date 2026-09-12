import { and, desc, eq } from "drizzle-orm";
import { getChatGPTUser } from "../../chatgpt-auth";
import { getDb } from "../../../db";
import { eventRevisions } from "../../../db/schema";

const STATUSES = new Set(["unreviewed", "gold", "rejected", "uncertain", "double_checked", "adjudicated"]);
const REVIEW_STATUSES = new Set(["double_checked", "adjudicated"]);
const HITTERS = new Set(["A", "B"]);
const LABEL_SOURCES = new Set(["machine_review", "manual"]);

type EventPayload = {
  video_id?: string;
  event_id?: string;
  candidate_frame?: number;
  candidate_time?: number;
  rally_id?: string;
  hitter?: string;
  stroke?: string;
  backhand?: boolean;
  aroundhead?: boolean;
  certainty?: string;
  notes?: string;
  review_note?: string;
  hit_position?: SpatialPoint | null;
  landing_position?: SpatialPoint | null;
  status?: string;
  label_source?: string;
  machine_prediction?: Record<string, unknown>;
};

type SpatialPoint = {
  image_x: number | null;
  image_y: number | null;
  court_x: number;
  court_y: number;
  source: "video" | "court";
};

function parseStoredPoint(value: string | null): SpatialPoint | null {
  if (!value) return null;
  try { return JSON.parse(value) as SpatialPoint; } catch { return null; }
}

function validSpatialPoint(value: SpatialPoint | null | undefined) {
  if (value == null) return true;
  const imagePairValid = (value.image_x === null && value.image_y === null)
    || (Number.isFinite(value.image_x) && Number.isFinite(value.image_y) && Number(value.image_x) >= 0 && Number(value.image_y) >= 0);
  return imagePairValid
    && Number.isFinite(value.court_x) && value.court_x >= 0 && value.court_x <= 6.1
    && Number.isFinite(value.court_y) && value.court_y >= 0 && value.court_y <= 13.4
    && (value.source === "video" || value.source === "court");
}

function serialize(row: typeof eventRevisions.$inferSelect) {
  return {
    id: row.id,
    video_id: row.videoId,
    event_id: row.eventId,
    revision: row.revision,
    candidate_frame: row.candidateFrame,
    candidate_time: row.candidateTimeMs / 1000,
    rally_id: row.rallyId,
    hitter: row.hitter,
    stroke: row.stroke,
    backhand: row.backhand,
    aroundhead: row.aroundhead,
    certainty: row.certainty,
    notes: row.notes,
    hit_position: parseStoredPoint(row.hitPositionJson),
    landing_position: parseStoredPoint(row.landingPositionJson),
    status: row.status,
    label_source: row.labelSource,
    machine_prediction: JSON.parse(row.machinePredictionJson) as Record<string, unknown>,
    annotator_id: row.annotatorId,
    reviewer_id: row.reviewerId,
    review_note: row.reviewNote,
    review_base_revision: row.reviewBaseRevision,
    reviewed_at: row.reviewedAt,
    created_at: row.createdAt,
  };
}

function routeError(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected event persistence error";
  const status = message.includes("no such table") ? 503 : 500;
  return Response.json(
    { error: status === 503 ? "标注数据库正在初始化，请稍后重试" : message },
    { status },
  );
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const videoId = url.searchParams.get("video_id")?.trim();
  const eventId = url.searchParams.get("event_id")?.trim();
  if (!videoId) return Response.json({ error: "video_id is required" }, { status: 400 });
  try {
    const db = getDb();
    if (eventId) {
      const history = await db.select().from(eventRevisions)
        .where(and(eq(eventRevisions.videoId, videoId), eq(eventRevisions.eventId, eventId)))
        .orderBy(desc(eventRevisions.revision));
      return Response.json({ revisions: history.map(serialize) });
    }
    const rows = await db.select().from(eventRevisions)
      .where(eq(eventRevisions.videoId, videoId))
      .orderBy(desc(eventRevisions.revision));
    const latest = new Map<string, typeof eventRevisions.$inferSelect>();
    for (const row of rows) if (!latest.has(row.eventId)) latest.set(row.eventId, row);
    return Response.json({ events: [...latest.values()].map(serialize) });
  } catch (error) {
    return routeError(error);
  }
}

export async function POST(request: Request) {
  try {
    const payload = await request.json() as EventPayload;
    const videoId = payload.video_id?.trim();
    const eventId = payload.event_id?.trim();
    const candidateTime = Number(payload.candidate_time);
    const candidateFrame = Math.round(Number(payload.candidate_frame));
    const hitter = payload.hitter?.trim() ?? "";
    const stroke = payload.stroke?.trim() ?? "";
    const status = payload.status?.trim() ?? "";
    const labelSource = payload.label_source?.trim() ?? "machine_review";
    if (!videoId || !eventId) return Response.json({ error: "video_id and event_id are required" }, { status: 400 });
    if (!Number.isFinite(candidateTime) || candidateTime < 0 || !Number.isInteger(candidateFrame) || candidateFrame < 0) {
      return Response.json({ error: "candidate time and frame must be non-negative" }, { status: 400 });
    }
    if (!HITTERS.has(hitter)) return Response.json({ error: "hitter must be A or B" }, { status: 400 });
    if (!stroke) return Response.json({ error: "stroke is required" }, { status: 400 });
    if (!STATUSES.has(status)) return Response.json({ error: "invalid status" }, { status: 400 });
    if (!LABEL_SOURCES.has(labelSource)) return Response.json({ error: "invalid label_source" }, { status: 400 });
    if (!validSpatialPoint(payload.hit_position) || !validSpatialPoint(payload.landing_position)) {
      return Response.json({ error: "hit_position and landing_position must be valid court coordinates" }, { status: 400 });
    }

    const db = getDb();
    const [latest] = await db.select({
      revision: eventRevisions.revision,
      status: eventRevisions.status,
      annotatorId: eventRevisions.annotatorId,
      reviewerId: eventRevisions.reviewerId,
      reviewNote: eventRevisions.reviewNote,
      reviewBaseRevision: eventRevisions.reviewBaseRevision,
      reviewedAt: eventRevisions.reviewedAt,
    }).from(eventRevisions)
      .where(and(eq(eventRevisions.videoId, videoId), eq(eventRevisions.eventId, eventId)))
      .orderBy(desc(eventRevisions.revision)).limit(1);
    const user = await getChatGPTUser();
    const actorId = user?.userId ?? (REVIEW_STATUSES.has(status) ? "public-reviewer" : "public-annotator");
    const isReviewStatus = REVIEW_STATUSES.has(status);
    const isReviewTransition = isReviewStatus && latest?.status !== status;
    if (latest?.status === "adjudicated" && status !== "adjudicated") {
      return Response.json({ error: "adjudicated events cannot be downgraded" }, { status: 409 });
    }
    if (latest?.status === "double_checked" && !["double_checked", "adjudicated"].includes(status)) {
      return Response.json({ error: "double-checked events can only advance to adjudicated" }, { status: 409 });
    }
    if (isReviewTransition && status === "double_checked" && !["gold", "uncertain"].includes(latest?.status ?? "")) {
      return Response.json({ error: "double check requires a Gold or uncertain revision" }, { status: 409 });
    }
    if (isReviewTransition && status === "adjudicated" && !["double_checked", "uncertain"].includes(latest?.status ?? "")) {
      return Response.json({ error: "adjudication requires a double-checked or uncertain revision" }, { status: 409 });
    }
    if (isReviewTransition && status === "double_checked" && user?.userId && latest?.annotatorId === user.userId) {
      return Response.json({ error: "复核需由另一位标注者完成" }, { status: 409 });
    }
    if (isReviewTransition && status === "adjudicated" && user?.userId && latest?.reviewerId === user.userId) {
      return Response.json({ error: "裁决需由另一位审核者完成" }, { status: 409 });
    }
    const createdAt = new Date().toISOString();
    const record: typeof eventRevisions.$inferInsert = {
      id: `ann_${crypto.randomUUID()}`,
      videoId,
      eventId,
      revision: (latest?.revision ?? 0) + 1,
      candidateFrame,
      candidateTimeMs: Math.round(candidateTime * 1000),
      rallyId: payload.rally_id?.trim() || "Model pilot",
      hitter,
      stroke,
      backhand: Boolean(payload.backhand),
      aroundhead: Boolean(payload.aroundhead),
      certainty: payload.certainty?.trim() || "确定",
      notes: payload.notes?.trim() || "",
      hitPositionJson: payload.hit_position ? JSON.stringify(payload.hit_position) : null,
      landingPositionJson: payload.landing_position ? JSON.stringify(payload.landing_position) : null,
      status,
      labelSource,
      machinePredictionJson: JSON.stringify(payload.machine_prediction ?? {}),
      annotatorId: isReviewStatus ? (latest?.annotatorId ?? actorId) : actorId,
      reviewerId: isReviewStatus ? (isReviewTransition ? actorId : latest?.reviewerId) : null,
      reviewNote: isReviewStatus ? (payload.review_note?.trim() || latest?.reviewNote || "") : null,
      reviewBaseRevision: isReviewStatus ? (isReviewTransition ? latest?.revision ?? null : latest?.reviewBaseRevision) : null,
      reviewedAt: isReviewStatus ? (isReviewTransition ? createdAt : latest?.reviewedAt) : null,
      createdAt,
    };
    const [saved] = await db.insert(eventRevisions).values(record).returning();
    return Response.json({ event: serialize(saved) }, { status: 201 });
  } catch (error) {
    return routeError(error);
  }
}
