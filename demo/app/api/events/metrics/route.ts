import { desc, eq } from "drizzle-orm";
import { getDb } from "../../../../db";
import { eventRevisions } from "../../../../db/schema";

function ratio(numerator: number, denominator: number) {
  return denominator > 0 ? numerator / denominator : null;
}

const GOLD_STATUSES = new Set(["gold", "double_checked", "adjudicated"]);

function routeError(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected Pilot metrics error";
  const status = message.includes("no such table") ? 503 : 500;
  return Response.json({ error: status === 503 ? "Pilot 数据库正在初始化，请稍后重试" : message }, { status });
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const videoId = url.searchParams.get("video_id")?.trim();
  const requestedTarget = Number(url.searchParams.get("target") ?? 50);
  const targetGold = Number.isInteger(requestedTarget) && requestedTarget > 0 ? Math.min(requestedTarget, 10000) : 50;
  if (!videoId) return Response.json({ error: "video_id is required" }, { status: 400 });

  try {
    const rows = await getDb().select().from(eventRevisions)
      .where(eq(eventRevisions.videoId, videoId))
      .orderBy(desc(eventRevisions.revision));
    const latest = new Map<string, typeof eventRevisions.$inferSelect>();
    for (const row of rows) if (!latest.has(row.eventId)) latest.set(row.eventId, row);
    const events = [...latest.values()];
    const gold = events.filter((event) => GOLD_STATUSES.has(event.status));
    const machineGold = gold.filter((event) => event.labelSource === "machine_review").length;
    const manualGold = gold.filter((event) => event.labelSource === "manual").length;
    const machineRejected = events.filter((event) => event.status === "rejected" && event.labelSource === "machine_review").length;
    const uncertain = events.filter((event) => event.status === "uncertain").length;
    const reviewed = events.filter((event) => event.status !== "unreviewed").length;

    return Response.json({
      metrics: {
        video_id: videoId,
        target_gold: targetGold,
        reviewed,
        gold: gold.length,
        machine_gold: machineGold,
        manual_gold: manualGold,
        machine_rejected: machineRejected,
        uncertain,
        annotated: events.filter((event) => event.status === "gold").length,
        double_checked: events.filter((event) => event.status === "double_checked").length,
        adjudicated: events.filter((event) => event.status === "adjudicated").length,
        total_revisions: rows.length,
        progress: Math.min(1, gold.length / targetGold),
        candidate_precision: ratio(machineGold, machineGold + machineRejected),
        estimated_recall: ratio(machineGold, machineGold + manualGold),
        recall_is_estimate: true,
        recall_assumption: "manual Gold events exhaustively represent missed machine candidates",
      },
    });
  } catch (error) {
    return routeError(error);
  }
}
