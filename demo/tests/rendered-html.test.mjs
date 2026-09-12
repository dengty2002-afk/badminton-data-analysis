import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the ShuttleLab annotation workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /ShuttleLab/);
  assert.match(html, /击球事件审核/);
  assert.match(html, /D1 已同步|读取标注/);
  assert.match(html, /在当前帧补录击球事件/);
  assert.match(html, /空间点位/);
  assert.match(html, /击球点/);
  assert.match(html, /落点/);
  assert.match(html, /Gold 复核/);
  assert.match(html, /复核通过/);
  assert.match(html, /裁决确认/);
  assert.match(html, /一致性/);
  assert.match(html, /D1 追加版本，不覆盖历史/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|Building your site/i);
});

test("keeps Gold annotations and player-side changes append-only and platform-backed", async () => {
  const [schema, route, metricsRoute, agreementRoute, sideRoute, historyModal, agreementModal, page, migration, sideMigration, pointMigration, reviewMigration] = await Promise.all([
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/events/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/events/metrics/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/events/agreement/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/player-sides/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/EventHistoryModal.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/AgreementModal.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0001_old_inhumans.sql", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0002_slippery_gambit.sql", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0003_parallel_charles_xavier.sql", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0004_mighty_captain_universe.sql", import.meta.url), "utf8"),
  ]);

  assert.match(schema, /eventRevisions/);
  assert.match(schema, /idx_event_revisions_video_event_revision/);
  assert.match(route, /revision:\s*\(latest\?\.revision \?\? 0\) \+ 1/);
  assert.match(route, /machinePredictionJson/);
  assert.match(route, /annotatorId/);
  assert.match(route, /hitPositionJson/);
  assert.match(route, /landingPositionJson/);
  assert.match(route, /validSpatialPoint/);
  assert.match(route, /REVIEW_STATUSES/);
  assert.match(route, /复核需由另一位标注者完成/);
  assert.match(route, /double-checked events can only advance to adjudicated/);
  assert.match(metricsRoute, /candidate_precision/);
  assert.match(metricsRoute, /estimated_recall/);
  assert.match(metricsRoute, /manual Gold events exhaustively represent missed machine candidates/);
  assert.match(metricsRoute, /double_checked/);
  assert.match(metricsRoute, /adjudicated/);
  assert.match(agreementRoute, /cohenKappa/);
  assert.match(agreementRoute, /hit_frame_within_3_agreement/);
  assert.match(agreementRoute, /reviewBaseRevision/);
  assert.match(agreementRoute, /disagreements/);
  assert.match(historyModal, /event_id=\$\{encodeURIComponent\(eventId\)\}/);
  assert.match(historyModal, /机器原预测/);
  assert.match(historyModal, /人工修订/);
  assert.match(migration, /CREATE TABLE `event_revisions`/);
  assert.match(migration, /CREATE UNIQUE INDEX `idx_event_revisions_video_event_revision`/);
  assert.match(schema, /playerSideAssignments/);
  assert.match(sideRoute, /effectiveFrame/);
  assert.match(sideRoute, /upperPlayer/);
  assert.match(sideRoute, /public-annotator/);
  assert.match(sideMigration, /CREATE TABLE `player_side_assignments`/);
  assert.match(sideMigration, /idx_player_sides_video_version/);
  assert.match(pointMigration, /ADD `hit_position_json` text/);
  assert.match(pointMigration, /ADD `landing_position_json` text/);
  assert.match(reviewMigration, /ADD `reviewer_id` text/);
  assert.match(reviewMigration, /ADD `review_base_revision` integer/);
  assert.match(page, /fetch\("\/api\/events"/);
  assert.match(page, /label_source/);
  assert.match(page, /shuttlelab_pilot_report_/);
  assert.match(page, /查看历史/);
  assert.match(page, /splitRallyBefore/);
  assert.match(page, /mergeWithPreviousRally/);
  assert.match(page, /recordSideSwitch/);
  assert.match(page, /hitter_side/);
  assert.match(page, /记录换边/);
  assert.match(page, /placePointOnVideo/);
  assert.match(page, /projectPoint/);
  assert.match(page, /hit_x,hit_y,landing_x,landing_y/);
  assert.match(historyModal, /revision\.landing_position/);
  assert.match(historyModal, /复核意见/);
  assert.match(page, /saveStatus\("double_checked"\)/);
  assert.match(page, /saveStatus\("adjudicated"\)/);
  assert.match(page, /setShowAgreement\(true\)/);
  assert.match(agreementModal, /Gold 标注一致性报告/);
  assert.match(agreementModal, /shuttlelab_agreement_/);
  assert.match(page, /已追加.*D1 修订/);
  assert.doesNotMatch(page, /BST-3 demo/);
  assert.doesNotMatch(page, /localStorage|sessionStorage/);
});
