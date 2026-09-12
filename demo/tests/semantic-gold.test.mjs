import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const catalog = JSON.parse(await readFile(new URL("../public/gold/catalog.json", import.meta.url), "utf8"));
const page = await readFile(new URL("../app/gold/page.tsx", import.meta.url), "utf8");
const api = await readFile(new URL("../app/api/semantic-gold/route.ts", import.meta.url), "utf8");
const schema = await readFile(new URL("../db/schema.ts", import.meta.url), "utf8");
const migration = await readFile(new URL("../drizzle/0005_lumpy_magus.sql", import.meta.url), "utf8");
const landingMigration = await readFile(new URL("../drizzle/0006_fuzzy_boom_boom.sql", import.meta.url), "utf8");

test("semantic Gold catalog contains the locked hit anchors", () => {
  assert.equal(catalog.schemaVersion, "shuttlelab-semantic-catalog-1");
  assert.equal(catalog.videoCount, 9);
  assert.equal(catalog.anchorCount, 337);
  const anchors = catalog.videos.flatMap((video) => {
    assert.ok(video.fps > 0);
    assert.equal(video.anchorCount, video.anchors.length);
    return video.anchors;
  });
  assert.equal(new Set(anchors.map((anchor) => anchor.anchorId)).size, anchors.length);
});

test("semantic Gold page exposes the frozen official vocabulary", () => {
  const types = [
    "net shot", "return net", "smash", "wrist smash", "lob", "defensive return lob",
    "clear", "drive", "driven flight", "back-court drive", "drop", "passive drop",
    "push", "rush", "defensive return drive", "cross-court net shot", "short service", "long service",
  ];
  for (const type of types) assert.match(page, new RegExp(`\\["${type.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"`));
  assert.match(page, /BLIND GOLD/);
  assert.match(page, /next_contact/);
  assert.match(page, /terminal/);
  assert.match(page, /landing_status/);
  assert.match(page, /landing_area/);
  assert.match(page, /shuttlelab_9_v2/);
  assert.match(page, /"out"/);
  assert.match(page, /caught/);
  assert.match(page, /unobservable/);
  assert.match(page, /semantic-area-specials/);
  assert.match(page, /setLandingStatus\("observed"\); setLandingArea\(id\)/);
  assert.doesNotMatch(page, /OUT_AREAS/);
  assert.doesNotMatch(page, /场外区域/);
  assert.doesNotMatch(page, /landing_x/);
  assert.doesNotMatch(page, /landing_y/);
  assert.doesNotMatch(page, /ground_contact/);
  assert.match(page, /webkitdirectory/);
  assert.match(page, /multiple/);
  assert.match(page, /视频只在本机读取，不会上传/);
});

test("landing area contract is explicit and rejects invented coordinates", () => {
  assert.match(api, /landingStatus === "observed"/);
  assert.match(api, /landingArea! < 1 \|\| landingArea! > 9/);
  assert.match(api, /landingStatus !== "observed" && landingArea != null/);
  assert.match(api, /landingX: null, landingY: null, landingFrame: null/);
  assert.match(api, /landingAreaSchema: "shuttlelab_9_v2"/);
  assert.match(schema, /landing_status/);
  assert.match(schema, /landing_area/);
  assert.match(schema, /landing_area_schema/);
});

test("semantic Gold migration creates query indexes and optimizes SQLite", () => {
  assert.match(migration, /CREATE TABLE `semantic_gold_revisions`/);
  assert.match(migration, /idx_semantic_gold_video_anchor_revision/);
  assert.match(migration, /idx_semantic_gold_video_status/);
  assert.match(migration, /PRAGMA optimize/);
  assert.match(landingMigration, /ADD `landing_status`/);
  assert.match(landingMigration, /ADD `landing_area` integer/);
  assert.match(landingMigration, /ADD `landing_area_schema`/);
  assert.match(landingMigration, /PRAGMA optimize/);
});
