import { asc, eq } from "drizzle-orm";
import { getDb } from "../../../../db";
import { eventRevisions } from "../../../../db/schema";

type Revision = typeof eventRevisions.$inferSelect;

function ratio(numerator: number, denominator: number) {
  return denominator > 0 ? numerator / denominator : null;
}

function cohenKappa(pairs: Array<[string, string]>) {
  if (!pairs.length) return null;
  const left = new Map<string, number>();
  const right = new Map<string, number>();
  let agreements = 0;
  for (const [first, second] of pairs) {
    if (first === second) agreements += 1;
    left.set(first, (left.get(first) ?? 0) + 1);
    right.set(second, (right.get(second) ?? 0) + 1);
  }
  const observed = agreements / pairs.length;
  const labels = new Set([...left.keys(), ...right.keys()]);
  const expected = [...labels].reduce((sum, label) => (
    sum + ((left.get(label) ?? 0) / pairs.length) * ((right.get(label) ?? 0) / pairs.length)
  ), 0);
  return Math.abs(1 - expected) < 1e-12 ? null : (observed - expected) / (1 - expected);
}

function routeError(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected agreement report error";
  const status = message.includes("no such table") || message.includes("no such column") ? 503 : 500;
  return Response.json({ error: status === 503 ? "一致性数据正在初始化，请稍后重试" : message }, { status });
}

export async function GET(request: Request) {
  const videoId = new URL(request.url).searchParams.get("video_id")?.trim();
  if (!videoId) return Response.json({ error: "video_id is required" }, { status: 400 });

  try {
    const rows = await getDb().select().from(eventRevisions)
      .where(eq(eventRevisions.videoId, videoId))
      .orderBy(asc(eventRevisions.eventId), asc(eventRevisions.revision));
    const byEvent = new Map<string, Revision[]>();
    for (const row of rows) byEvent.set(row.eventId, [...(byEvent.get(row.eventId) ?? []), row]);

    const comparisons: Array<{ base: Revision; review: Revision }> = [];
    let adjudicatedCount = 0;
    for (const revisions of byEvent.values()) {
      if (revisions.at(-1)?.status === "adjudicated") adjudicatedCount += 1;
      const review = [...revisions].reverse().find((item) => item.status === "double_checked" && item.reviewBaseRevision != null)
        ?? [...revisions].reverse().find((item) => item.status === "adjudicated" && item.reviewBaseRevision != null);
      if (!review) continue;
      const base = revisions.find((item) => item.revision === review.reviewBaseRevision);
      if (base) comparisons.push({ base, review });
    }

    const hitterMatches = comparisons.filter(({ base, review }) => base.hitter === review.hitter).length;
    const frameMatches = comparisons.filter(({ base, review }) => Math.abs(base.candidateFrame - review.candidateFrame) <= 3).length;
    const strokeMatches = comparisons.filter(({ base, review }) => base.stroke === review.stroke).length;
    const strokePairs = comparisons.map(({ base, review }) => [base.stroke, review.stroke] as [string, string]);
    const disagreements = comparisons.flatMap(({ base, review }) => {
      const changedFields = [
        base.hitter !== review.hitter ? "hitter" : null,
        Math.abs(base.candidateFrame - review.candidateFrame) > 3 ? "hit_frame" : null,
        base.stroke !== review.stroke ? "stroke" : null,
      ].filter((value): value is string => Boolean(value));
      if (!changedFields.length) return [];
      return [{
        event_id: review.eventId,
        base_revision: base.revision,
        review_revision: review.revision,
        changed_fields: changedFields,
        original: { hitter: base.hitter, hit_frame: base.candidateFrame, stroke: base.stroke },
        review: { hitter: review.hitter, hit_frame: review.candidateFrame, stroke: review.stroke },
        reviewer_id: review.reviewerId,
        review_note: review.reviewNote,
      }];
    });

    return Response.json({
      report: {
        video_id: videoId,
        generated_at: new Date().toISOString(),
        sample_size: comparisons.length,
        adjudicated_count: adjudicatedCount,
        hitter_agreement: ratio(hitterMatches, comparisons.length),
        hit_frame_within_3_agreement: ratio(frameMatches, comparisons.length),
        stroke_agreement: ratio(strokeMatches, comparisons.length),
        stroke_cohen_kappa: cohenKappa(strokePairs),
        disagreements_count: disagreements.length,
        targets: { hitter_agreement: 0.98, hit_frame_within_3_agreement: 0.95, stroke_cohen_kappa: 0.8 },
        disagreements,
      },
    });
  } catch (error) {
    return routeError(error);
  }
}
