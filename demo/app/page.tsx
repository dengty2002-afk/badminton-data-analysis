"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { CalibrationModal } from "./CalibrationModal";
import { EventHistoryModal } from "./EventHistoryModal";
import { AgreementModal } from "./AgreementModal";
import { projectPoint } from "../lib/homography";

type EventStatus = "unreviewed" | "gold" | "rejected" | "uncertain" | "double_checked" | "adjudicated";
type SpatialPoint = {
  image_x: number | null;
  image_y: number | null;
  court_x: number;
  court_y: number;
  source: "video" | "court";
};
type HitEvent = {
  id: string;
  rally: string;
  time: number;
  hitter: "A" | "B";
  stroke: string;
  confidence: number;
  trajectory: number;
  wrist: number;
  pose: number;
  status: EventStatus;
  revision?: number;
  labelSource?: "machine_review" | "manual";
  backhand?: boolean;
  aroundhead?: boolean;
  certainty?: string;
  notes?: string;
  machineHitter?: "A" | "B";
  machineStroke?: string;
  hitterSide?: "upper" | "lower";
  hitPosition?: SpatialPoint | null;
  landingPosition?: SpatialPoint | null;
  reviewerId?: string | null;
  reviewNote?: string | null;
  reviewBaseRevision?: number | null;
  reviewedAt?: string | null;
};

type PersistedEvent = {
  event_id: string;
  revision: number;
  candidate_frame: number;
  candidate_time: number;
  rally_id: string;
  hitter: "A" | "B";
  stroke: string;
  backhand: boolean;
  aroundhead: boolean;
  certainty: string;
  notes: string;
  hit_position: SpatialPoint | null;
  landing_position: SpatialPoint | null;
  status: EventStatus;
  label_source: "machine_review" | "manual";
  machine_prediction: Partial<Pick<HitEvent, "confidence" | "trajectory" | "wrist" | "pose">> & { hitter?: "A" | "B"; stroke?: string; side?: "upper" | "lower" };
  reviewer_id: string | null;
  review_note: string | null;
  review_base_revision: number | null;
  reviewed_at: string | null;
};

type CalibrationRecord = {
  version: number;
  homography: number[][];
  quality_status: string;
};

type SideAssignment = {
  id: string;
  version: number;
  effective_frame: number;
  effective_time: number;
  upper_player: "A" | "B";
  lower_player: "A" | "B";
  source: string;
  editor_id: string;
  created_at: string;
};

type VideoManifest = {
  schema_version: string;
  generated_at: string;
  pipeline_version: string;
  video: {
    video_id: string;
    file_name: string;
    checksum: string;
    file_size_bytes: number;
    duration_sec: number;
    width: number;
    height: number;
    fps: number;
    frame_count: number;
    has_audio: boolean;
    match_id: string;
    domain: string;
    redistribution: string;
    status: string;
  };
  latest_run: { run_id: string; status: string; completed_at: string };
};

type ModelCandidateFeed = {
  schema_version: string;
  pipeline_version: string;
  events: Array<HitEvent & { hitter_side?: "upper" | "lower" }>;
};

type PilotMetrics = {
  target_gold: number;
  reviewed: number;
  gold: number;
  machine_gold: number;
  manual_gold: number;
  machine_rejected: number;
  uncertain: number;
  total_revisions: number;
  progress: number;
  candidate_precision: number | null;
  estimated_recall: number | null;
  recall_is_estimate: boolean;
  annotated: number;
  double_checked: number;
  adjudicated: number;
};

const STROKES = [
  "短发球",
  "长发球",
  "高远球",
  "吊球",
  "杀球",
  "挑球",
  "平抽 / 推球",
  "网前球",
  "防守回球",
];

const INITIAL_EVENTS: HitEvent[] = [
  { id: "EVT-0018", rally: "Rally 03", time: 5.36, hitter: "B", stroke: "短发球", confidence: 0.94, trajectory: 0.92, wrist: 0.88, pose: 0.91, status: "gold" },
  { id: "EVT-0019", rally: "Rally 03", time: 7.12, hitter: "A", stroke: "挑球", confidence: 0.89, trajectory: 0.86, wrist: 0.91, pose: 0.83, status: "gold" },
  { id: "EVT-0020", rally: "Rally 03", time: 9.84, hitter: "B", stroke: "高远球", confidence: 0.91, trajectory: 0.93, wrist: 0.87, pose: 0.86, status: "gold" },
  { id: "EVT-0021", rally: "Rally 03", time: 12.68, hitter: "A", stroke: "杀球", confidence: 0.87, trajectory: 0.94, wrist: 0.81, pose: 0.85, status: "unreviewed" },
  { id: "EVT-0022", rally: "Rally 03", time: 14.44, hitter: "B", stroke: "防守回球", confidence: 0.73, trajectory: 0.78, wrist: 0.69, pose: 0.71, status: "unreviewed" },
  { id: "EVT-0023", rally: "Rally 03", time: 16.92, hitter: "A", stroke: "网前球", confidence: 0.81, trajectory: 0.84, wrist: 0.76, pose: 0.79, status: "unreviewed" },
  { id: "EVT-0024", rally: "Rally 04", time: 21.16, hitter: "B", stroke: "长发球", confidence: 0.96, trajectory: 0.95, wrist: 0.92, pose: 0.93, status: "unreviewed" },
  { id: "EVT-0025", rally: "Rally 04", time: 24.08, hitter: "A", stroke: "吊球", confidence: 0.68, trajectory: 0.72, wrist: 0.61, pose: 0.70, status: "unreviewed" },
  { id: "EVT-0026", rally: "Rally 04", time: 26.40, hitter: "B", stroke: "平抽 / 推球", confidence: 0.84, trajectory: 0.88, wrist: 0.79, pose: 0.82, status: "unreviewed" },
];

function mergePersistedEvent(review: PersistedEvent, base?: HitEvent): HitEvent {
  const machine = review.machine_prediction ?? {};
  return {
    id: review.event_id,
    rally: review.rally_id === "Model pilot" && base?.rally ? base.rally : review.rally_id,
    time: review.candidate_time,
    hitter: review.hitter,
    stroke: review.stroke,
    confidence: base?.confidence ?? (Number(machine.confidence) || 0),
    trajectory: base?.trajectory ?? (Number(machine.trajectory) || 0),
    wrist: base?.wrist ?? (Number(machine.wrist) || 0),
    pose: base?.pose ?? (Number(machine.pose) || 0),
    status: review.status,
    revision: review.revision,
    labelSource: review.label_source,
    backhand: review.backhand,
    aroundhead: review.aroundhead,
    certainty: review.certainty,
    notes: review.notes,
    hitPosition: review.hit_position,
    landingPosition: review.landing_position,
    reviewerId: review.reviewer_id,
    reviewNote: review.review_note,
    reviewBaseRevision: review.review_base_revision,
    reviewedAt: review.reviewed_at,
    machineHitter: base?.machineHitter ?? review.machine_prediction.hitter,
    machineStroke: base?.machineStroke ?? review.machine_prediction.stroke,
    hitterSide: base?.hitterSide ?? review.machine_prediction.side,
  };
}

function sideAssignmentAtFrame(assignments: SideAssignment[], frame: number): SideAssignment {
  let current: SideAssignment = {
    id: "side_default",
    version: 0,
    effective_frame: 0,
    effective_time: 0,
    upper_player: "A",
    lower_player: "B",
    source: "default",
    editor_id: "system",
    created_at: "",
  };
  for (const assignment of assignments) {
    if (assignment.effective_frame <= frame) current = assignment;
    else break;
  }
  return current;
}

function playerForSide(side: "upper" | "lower" | undefined, frame: number, assignments: SideAssignment[]) {
  if (!side) return undefined;
  const assignment = sideAssignmentAtFrame(assignments, frame);
  return side === "lower" ? assignment.lower_player : assignment.upper_player;
}

function sameSpatialPoint(left: SpatialPoint | null | undefined, right: SpatialPoint | null | undefined) {
  return JSON.stringify(left ?? null) === JSON.stringify(right ?? null);
}

function renumberRallies(events: HitEvent[]) {
  const ordered = [...events].sort((a, b) => a.time - b.time);
  const names = new Map<string, string>();
  for (const event of ordered) {
    if (!names.has(event.rally)) names.set(event.rally, `Rally ${String(names.size + 1).padStart(2, "0")}`);
  }
  return ordered.map((event) => ({ ...event, rally: names.get(event.rally)! }));
}

function splitRallyBefore(events: HitEvent[], eventId: string) {
  const ordered = [...events].sort((a, b) => a.time - b.time);
  const index = ordered.findIndex((event) => event.id === eventId);
  if (index <= 0 || ordered[index - 1].rally !== ordered[index].rally) return events;
  const rally = ordered[index].rally;
  const temporary = `${rally}__split_${eventId}`;
  return renumberRallies(ordered.map((event, eventIndex) => (
    eventIndex >= index && event.rally === rally ? { ...event, rally: temporary } : event
  )));
}

function mergeWithPreviousRally(events: HitEvent[], rally: string) {
  const ordered = [...events].sort((a, b) => a.time - b.time);
  const rallyOrder = [...new Set(ordered.map((event) => event.rally))];
  const rallyIndex = rallyOrder.indexOf(rally);
  if (rallyIndex <= 0) return events;
  const previous = rallyOrder[rallyIndex - 1];
  return renumberRallies(ordered.map((event) => event.rally === rally ? { ...event, rally: previous } : event));
}

function formatTime(seconds: number) {
  const safe = Math.max(0, seconds || 0);
  const min = Math.floor(safe / 60);
  const sec = Math.floor(safe % 60);
  const ms = Math.floor((safe % 1) * 100);
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${String(ms).padStart(2, "0")}`;
}

function Icon({ children }: { children: React.ReactNode }) {
  return <span className="nav-icon" aria-hidden="true">{children}</span>;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [events, setEvents] = useState<HitEvent[]>(INITIAL_EVENTS);
  const [activeId, setActiveId] = useState("EVT-0021");
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(12.68);
  const [duration, setDuration] = useState(0);
  const [manifest, setManifest] = useState<VideoManifest | null>(null);
  const [showPose, setShowPose] = useState(true);
  const [showTrail, setShowTrail] = useState(true);
  const [hitter, setHitter] = useState<"A" | "B">("A");
  const [stroke, setStroke] = useState("杀球");
  const [backhand, setBackhand] = useState(false);
  const [aroundhead, setAroundhead] = useState(false);
  const [certainty, setCertainty] = useState("确定");
  const [notes, setNotes] = useState("");
  const [toast, setToast] = useState("");
  const [queueFilter, setQueueFilter] = useState<"all" | "pending" | "review">("all");
  const [showCalibration, setShowCalibration] = useState(false);
  const [syncState, setSyncState] = useState<"loading" | "synced" | "saving" | "error">("loading");
  const [pilotMetrics, setPilotMetrics] = useState<PilotMetrics | null>(null);
  const [historyEventId, setHistoryEventId] = useState<string | null>(null);
  const [showAgreement, setShowAgreement] = useState(false);
  const [sideAssignments, setSideAssignments] = useState<SideAssignment[]>([]);
  const [calibration, setCalibration] = useState<CalibrationRecord | null>(null);
  const [pointMode, setPointMode] = useState<"hit" | "landing" | null>(null);
  const [hitPosition, setHitPosition] = useState<SpatialPoint | null>(null);
  const [landingPosition, setLandingPosition] = useState<SpatialPoint | null>(null);
  const [reviewNote, setReviewNote] = useState("");

  const active = events.find((event) => event.id === activeId) ?? events[0];
  const activeIndex = events.findIndex((event) => event.id === active.id);
  const goldCount = events.filter((event) => ["gold", "double_checked", "adjudicated"].includes(event.status)).length;
  const reviewQueue = events.filter((event) => ["gold", "uncertain", "double_checked"].includes(event.status));
  const visibleEvents = queueFilter === "pending"
    ? events.filter((event) => event.status === "unreviewed")
    : queueFilter === "review" ? reviewQueue : events;
  const fps = manifest?.video.fps ?? 30;
  const videoId = manifest?.video.video_id;
  const orderedRallies = [...new Set(events.map((event) => event.rally))];
  const activeRallyMembers = events.filter((event) => event.rally === active.rally).sort((a, b) => a.time - b.time);
  const canSplitRally = activeRallyMembers.findIndex((event) => event.id === active.id) > 0;
  const canMergeRally = orderedRallies.indexOf(active.rally) > 0;
  const isReviewLocked = active.status === "double_checked" || active.status === "adjudicated";
  const draftDirty = active.hitter !== hitter
    || active.stroke !== stroke
    || Math.abs(active.time - currentTime) > 1 / Math.max(1, fps * 2)
    || Boolean(active.backhand) !== backhand
    || Boolean(active.aroundhead) !== aroundhead
    || (active.certainty ?? "确定") !== certainty
    || (active.notes ?? "") !== notes
    || !sameSpatialPoint(active.hitPosition, hitPosition)
    || !sameSpatialPoint(active.landingPosition, landingPosition)
    || (active.reviewNote ?? "") !== reviewNote;

  /* Event selection intentionally resets the editable annotation draft. */
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    fetch("/data/video-manifest.json")
      .then((response) => {
        if (!response.ok) throw new Error("manifest unavailable");
        return response.json() as Promise<VideoManifest>;
      })
      .then(setManifest)
      .catch(() => setManifest(null));
  }, []);

  useEffect(() => {
    if (!videoId) return;
    const controller = new AbortController();
    const load = async () => {
      setSyncState("loading");
      try {
        const feedResponse = await fetch("/data/model-candidates.json", { signal: controller.signal });
        if (!feedResponse.ok) throw new Error("model candidates unavailable");
        const feed = await feedResponse.json() as ModelCandidateFeed;
        let assignments: SideAssignment[] = [];
        const sidesResponse = await fetch(`/api/player-sides?video_id=${encodeURIComponent(videoId)}`, { signal: controller.signal });
        if (sidesResponse.ok) {
          const sidesPayload = await sidesResponse.json() as { assignments: SideAssignment[] };
          assignments = sidesPayload.assignments;
          setSideAssignments(assignments);
        }
        const calibrationResponse = await fetch(`/api/calibrations?video_id=${encodeURIComponent(videoId)}`, { signal: controller.signal });
        if (calibrationResponse.ok) {
          const calibrationPayload = await calibrationResponse.json() as { calibrations: CalibrationRecord[] };
          setCalibration(calibrationPayload.calibrations.find((item) => item.quality_status === "accepted") ?? null);
        }
        const feedEvents = feed.events.map((event) => {
          const resolvedHitter = playerForSide(event.hitter_side, Math.round(event.time * fps), assignments) ?? event.hitter;
          return {
            ...event,
            hitter: resolvedHitter,
            labelSource: "machine_review" as const,
            machineHitter: resolvedHitter,
            machineStroke: event.stroke,
            hitterSide: event.hitter_side,
          };
        });
        if (feedEvents.length) {
          setEvents(feedEvents);
          setActiveId(feedEvents[0].id);
        }
        const annotationsResponse = await fetch(`/api/events?video_id=${encodeURIComponent(videoId)}`, { signal: controller.signal });
        if (!annotationsResponse.ok) throw new Error("D1 annotations unavailable");
        const payload = await annotationsResponse.json() as { events: PersistedEvent[] };
        const bases = new Map(feedEvents.map((event) => [event.id, event]));
        const reviewed = new Map(payload.events.map((review) => [review.event_id, mergePersistedEvent(review, bases.get(review.event_id))]));
        const merged = feedEvents.map((event) => reviewed.get(event.id) ?? event);
        for (const review of payload.events) {
          if (!bases.has(review.event_id)) merged.push(reviewed.get(review.event_id)!);
        }
        if (merged.length) {
          setEvents(merged);
          setActiveId((current) => merged.some((event) => event.id === current) ? current : merged[0].id);
        }
        const metricsResponse = await fetch(`/api/events/metrics?video_id=${encodeURIComponent(videoId)}&target=50`, { signal: controller.signal });
        if (metricsResponse.ok) {
          const metricsPayload = await metricsResponse.json() as { metrics: PilotMetrics };
          setPilotMetrics(metricsPayload.metrics);
        }
        setSyncState("synced");
      } catch (error) {
        if ((error as Error).name !== "AbortError") setSyncState("error");
      }
    };
    void load();
    return () => controller.abort();
  }, [fps, videoId]);

  useEffect(() => {
    setHitter(active.hitter);
    setStroke(active.stroke);
    setCurrentTime(active.time);
    setBackhand(Boolean(active.backhand));
    setAroundhead(Boolean(active.aroundhead));
    setCertainty(active.certainty ?? "确定");
    setNotes(active.notes ?? "");
    setHitPosition(active.hitPosition ?? null);
    setLandingPosition(active.landingPosition ?? null);
    setReviewNote(active.reviewNote ?? "");
    setPointMode(null);
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.currentTime = Math.min(active.time, Math.max(0, (video.duration || active.time + 1) - 0.1));
    }
    setPlaying(false);
  }, [active.aroundhead, active.backhand, active.certainty, active.hitPosition, active.hitter, active.id, active.landingPosition, active.notes, active.reviewNote, active.stroke, active.time]);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2400);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const seek = useCallback((time: number) => {
    const video = videoRef.current;
    if (!video) return;
    const next = Math.max(0, Math.min(time, Math.max(0, video.duration || duration || time)));
    video.currentTime = next;
    setCurrentTime(next);
  }, [duration]);

  const stepFrames = useCallback((count: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    setPlaying(false);
    seek(video.currentTime + count / fps);
  }, [fps, seek]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().catch(() => undefined);
      setPlaying(true);
    } else {
      video.pause();
      setPlaying(false);
    }
  }, []);

  const goNext = useCallback(() => {
    const next = events.slice(activeIndex + 1).find((event) => event.status === "unreviewed") ?? events[(activeIndex + 1) % events.length];
    setActiveId(next.id);
  }, [activeIndex, events]);

  const goNextReview = useCallback(() => {
    const candidates = events.filter((event) => event.id !== active.id && ["gold", "uncertain", "double_checked"].includes(event.status));
    const next = candidates.find((event) => event.time > active.time) ?? candidates[0];
    if (next) setActiveId(next.id); else goNext();
  }, [active.id, active.time, events, goNext]);

  const saveStatus = useCallback(async (status: EventStatus) => {
    if (!videoId || syncState === "saving") return;
    setSyncState("saving");
    const nextReviewNote = ["double_checked", "adjudicated"].includes(status) ? reviewNote : null;
    const nextEvent: HitEvent = {
      ...active,
      hitter,
      stroke,
      time: currentTime,
      status,
      backhand,
      aroundhead,
      certainty,
      notes,
      hitPosition,
      landingPosition,
      reviewNote: nextReviewNote,
      confidence: status === "uncertain" ? Math.min(active.confidence, 0.59) : active.confidence,
    };
    try {
      const response = await fetch("/api/events", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          video_id: videoId,
          event_id: active.id,
          candidate_frame: Math.round(currentTime * fps),
          candidate_time: currentTime,
          rally_id: active.rally,
          hitter,
          stroke,
          backhand,
          aroundhead,
          certainty,
          notes,
          hit_position: hitPosition,
          landing_position: landingPosition,
          review_note: nextReviewNote,
          status,
          label_source: active.labelSource ?? "machine_review",
          machine_prediction: {
            hitter: active.machineHitter,
            stroke: active.machineStroke,
            side: active.hitterSide,
            confidence: active.confidence,
            trajectory: active.trajectory,
            wrist: active.wrist,
            pose: active.pose,
          },
        }),
      });
      const payload = await response.json() as { event?: PersistedEvent; error?: string };
      if (!response.ok || !payload.event) throw new Error(payload.error || "保存失败");
      const saved = mergePersistedEvent(payload.event, nextEvent);
      setEvents((current) => current.map((event) => event.id === active.id ? saved : event));
      setSyncState("synced");
      void fetch(`/api/events/metrics?video_id=${encodeURIComponent(videoId)}&target=50`)
        .then((response) => response.ok ? response.json() as Promise<{ metrics: PilotMetrics }> : null)
        .then((payload) => payload && setPilotMetrics(payload.metrics))
        .catch(() => undefined);
      setToast(status === "gold"
        ? `${active.id} 已写入 Gold · 版本 ${saved.revision}`
        : status === "double_checked"
          ? `${active.id} 已完成独立复核 · 版本 ${saved.revision}`
          : status === "adjudicated"
            ? `${active.id} 已完成分歧裁决 · 版本 ${saved.revision}`
        : status === "rejected"
          ? `${active.id} 已标记为误报 · 版本 ${saved.revision}`
          : status === "unreviewed"
            ? `${active.id} 已恢复到候选队列`
            : `${active.id} 已进入复核队列 · 版本 ${saved.revision}`);
      if (status !== "unreviewed") window.setTimeout(["double_checked", "adjudicated"].includes(status) ? goNextReview : goNext, 360);
    } catch (error) {
      setSyncState("error");
      setToast(error instanceof Error ? error.message : "D1 保存失败，请重试");
    }
  }, [active, aroundhead, backhand, certainty, currentTime, fps, goNext, goNextReview, hitPosition, hitter, landingPosition, notes, reviewNote, stroke, syncState, videoId]);

  const addManualEvent = useCallback(() => {
    const id = `EVT-U${Date.now().toString(36).toUpperCase()}`;
    const manual: HitEvent = {
      id,
      rally: "人工补录",
      time: currentTime,
      hitter,
      stroke: "待标注",
      confidence: 0,
      trajectory: 0,
      wrist: 0,
      pose: 0,
      status: "unreviewed",
      labelSource: "manual",
      certainty: "确定",
    };
    setEvents((current) => [...current, manual].sort((a, b) => a.time - b.time));
    setActiveId(id);
    setToast(`${id} 已在当前帧创建，完成字段后保存即可`);
  }, [currentTime, hitter]);

  const persistRallyChanges = useCallback(async (nextEvents: HitEvent[], message: string) => {
    if (!videoId || syncState === "saving") return;
    const currentById = new Map(events.map((event) => [event.id, event]));
    const changed = nextEvents.filter((event) => currentById.get(event.id)?.rally !== event.rally);
    if (!changed.length) return;
    setSyncState("saving");
    try {
      const saved = await Promise.all(changed.map(async (event) => {
        const response = await fetch("/api/events", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            video_id: videoId,
            event_id: event.id,
            candidate_frame: Math.round(event.time * fps),
            candidate_time: event.time,
            rally_id: event.rally,
            hitter: event.hitter,
            stroke: event.stroke,
            backhand: Boolean(event.backhand),
            aroundhead: Boolean(event.aroundhead),
            certainty: event.certainty ?? "确定",
            notes: event.notes ?? "",
            hit_position: event.hitPosition ?? null,
            landing_position: event.landingPosition ?? null,
            review_note: event.reviewNote ?? null,
            status: event.status,
            label_source: event.labelSource ?? "machine_review",
            machine_prediction: {
              hitter: event.machineHitter,
              stroke: event.machineStroke,
              side: event.hitterSide,
              confidence: event.confidence,
              trajectory: event.trajectory,
              wrist: event.wrist,
              pose: event.pose,
            },
          }),
        });
        const payload = await response.json() as { event?: PersistedEvent; error?: string };
        if (!response.ok || !payload.event) throw new Error(payload.error || "回合保存失败");
        return mergePersistedEvent(payload.event, event);
      }));
      const savedById = new Map(saved.map((event) => [event.id, event]));
      setEvents(nextEvents.map((event) => savedById.get(event.id) ?? event));
      setSyncState("synced");
      void fetch(`/api/events/metrics?video_id=${encodeURIComponent(videoId)}&target=50`)
        .then((response) => response.ok ? response.json() as Promise<{ metrics: PilotMetrics }> : null)
        .then((payload) => payload && setPilotMetrics(payload.metrics))
        .catch(() => undefined);
      setToast(`${message}，已追加 ${saved.length} 条 D1 修订`);
    } catch (error) {
      setSyncState("error");
      setToast(error instanceof Error ? error.message : "回合保存失败，请刷新后重试");
    }
  }, [events, fps, syncState, videoId]);

  const recordSideSwitch = useCallback(async () => {
    if (!videoId || syncState === "saving" || draftDirty) return;
    const effectiveFrame = Math.round(currentTime * fps);
    const current = sideAssignmentAtFrame(sideAssignments, effectiveFrame);
    setSyncState("saving");
    try {
      const response = await fetch("/api/player-sides", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          video_id: videoId,
          effective_frame: effectiveFrame,
          effective_time: currentTime,
          upper_player: current.lower_player,
          lower_player: current.upper_player,
        }),
      });
      const payload = await response.json() as { assignment?: SideAssignment; error?: string };
      if (!response.ok || !payload.assignment) throw new Error(payload.error || "换边保存失败");
      const nextAssignments = [...sideAssignments, payload.assignment]
        .sort((a, b) => a.effective_frame - b.effective_frame || a.version - b.version);
      setSideAssignments(nextAssignments);
      setEvents((currentEvents) => currentEvents.map((event) => {
        if (event.status !== "unreviewed" || event.labelSource !== "machine_review" || !event.hitterSide) return event;
        const mapped = playerForSide(event.hitterSide, Math.round(event.time * fps), nextAssignments);
        return mapped ? { ...event, hitter: mapped, machineHitter: mapped } : event;
      }));
      setSyncState("synced");
      setToast(`已从第 ${effectiveFrame} 帧记录换边 · 远场 ${payload.assignment.upper_player} / 近场 ${payload.assignment.lower_player}`);
    } catch (error) {
      setSyncState("error");
      setToast(error instanceof Error ? error.message : "换边保存失败，请重试");
    }
  }, [currentTime, draftDirty, fps, sideAssignments, syncState, videoId]);

  const placeSpatialPoint = useCallback((point: SpatialPoint) => {
    if (pointMode === "hit") setHitPosition(point);
    if (pointMode === "landing") setLandingPosition(point);
    setToast(`${pointMode === "hit" ? "击球点" : "落点"}已更新 · (${point.court_x.toFixed(2)}, ${point.court_y.toFixed(2)}m)`);
    setPointMode(null);
  }, [pointMode]);

  const placePointOnVideo = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (!pointMode || !manifest || !calibration) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const videoRatio = manifest.video.width / manifest.video.height;
    const renderedWidth = Math.min(rect.width, rect.height * videoRatio);
    const renderedHeight = renderedWidth / videoRatio;
    const offsetX = (rect.width - renderedWidth) / 2;
    const offsetY = (rect.height - renderedHeight) / 2;
    const normalizedX = (event.clientX - rect.left - offsetX) / renderedWidth;
    const normalizedY = (event.clientY - rect.top - offsetY) / renderedHeight;
    if (normalizedX < 0 || normalizedX > 1 || normalizedY < 0 || normalizedY > 1) return;
    const image = { x: normalizedX * manifest.video.width, y: normalizedY * manifest.video.height };
    const court = projectPoint(calibration.homography, image);
    if (!Number.isFinite(court.x) || !Number.isFinite(court.y) || court.x < 0 || court.x > 6.1 || court.y < 0 || court.y > 13.4) {
      setToast("点击位置不在已标定球场内，请重新选择");
      return;
    }
    videoRef.current?.pause();
    placeSpatialPoint({
      image_x: Math.round(image.x * 10) / 10,
      image_y: Math.round(image.y * 10) / 10,
      court_x: Math.round(court.x * 1000) / 1000,
      court_y: Math.round(court.y * 1000) / 1000,
      source: "video",
    });
  }, [calibration, manifest, placeSpatialPoint, pointMode]);

  const placePointOnCourt = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (!pointMode) return;
    const rect = event.currentTarget.getBoundingClientRect();
    placeSpatialPoint({
      image_x: null,
      image_y: null,
      court_x: Math.round(Math.max(0, Math.min(6.1, ((event.clientX - rect.left) / rect.width) * 6.1)) * 1000) / 1000,
      court_y: Math.round(Math.max(0, Math.min(13.4, ((event.clientY - rect.top) / rect.height) * 13.4)) * 1000) / 1000,
      source: "court",
    });
  }, [placeSpatialPoint, pointMode]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (["INPUT", "TEXTAREA"].includes(target.tagName)) return;
      if (event.code === "Space") { event.preventDefault(); togglePlay(); }
      if (event.key === "ArrowLeft") { event.preventDefault(); stepFrames(event.shiftKey ? -5 : -1); }
      if (event.key === "ArrowRight") { event.preventDefault(); stepFrames(event.shiftKey ? 5 : 1); }
      if (event.key.toLowerCase() === "a") setHitter("A");
      if (event.key.toLowerCase() === "b") setHitter("B");
      if (/^[1-9]$/.test(event.key)) setStroke(STROKES[Number(event.key) - 1]);
      if (event.key === "Enter" && !isReviewLocked) void saveStatus("gold");
      if (event.key.toLowerCase() === "x" && !isReviewLocked) void saveStatus("rejected");
      if (event.key.toLowerCase() === "u" && !isReviewLocked) void saveStatus("uncertain");
      if (event.key.toLowerCase() === "r" && active.status === "rejected") void saveStatus("unreviewed");
      if (event.key.toLowerCase() === "h") setPointMode("hit");
      if (event.key.toLowerCase() === "l") setPointMode("landing");
      if (event.key === "Escape") setPointMode(null);
      if (event.key.toLowerCase() === "n") goNext();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active.status, goNext, isReviewLocked, saveStatus, stepFrames, togglePlay]);

  const exportCsv = () => {
    const header = "event_id,rally_id,hit_time,player,stroke_type,stroke_type_confidence,hit_x,hit_y,landing_x,landing_y,label_source,annotation_status,model_version";
    const rows = events.map((event) => [event.id, event.rally.replace(" ", "_"), event.time.toFixed(3), event.hitter, event.stroke, event.confidence.toFixed(2), event.hitPosition?.court_x ?? "", event.hitPosition?.court_y ?? "", event.landingPosition?.court_x ?? "", event.landingPosition?.court_y ?? "", event.labelSource ?? "machine_review", event.status, "hit-rules-0.2.0"].join(","));
    const blob = new Blob(["\ufeff" + [header, ...rows].join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "shuttlelab_demo_dataset.csv";
    link.click();
    URL.revokeObjectURL(url);
    setToast("已导出 ShuttleSet-like CSV");
  };

  const exportPilotReport = () => {
    const report = {
      schema_version: "1.0",
      generated_at: new Date().toISOString(),
      video: manifest?.video ?? null,
      metrics: pilotMetrics,
      recall_note: "estimated_recall is valid only when manual Gold exhaustively captures missed machine candidates",
      events: events.map((event) => ({
        event_id: event.id,
        rally_id: event.rally,
        hit_time: event.time,
        hit_frame: Math.round(event.time * fps),
        hitter: event.hitter,
        stroke: event.stroke,
        status: event.status,
        revision: event.revision ?? 0,
        label_source: event.labelSource ?? "machine_review",
        reviewer_id: event.reviewerId ?? null,
        review_note: event.reviewNote ?? null,
        review_base_revision: event.reviewBaseRevision ?? null,
        reviewed_at: event.reviewedAt ?? null,
        hit_position: event.hitPosition ?? null,
        landing_position: event.landingPosition ?? null,
        machine_prediction: {
          hitter: event.machineHitter ?? null,
          stroke: event.machineStroke ?? null,
          side: event.hitterSide ?? null,
          confidence: event.confidence,
        },
      })),
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `shuttlelab_pilot_report_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
    setToast("已导出 Pilot 评估报告");
  };

  const frame = Math.round(currentTime * fps);
  const currentSide = sideAssignmentAtFrame(sideAssignments, frame);
  const eventDelta = Math.round((currentTime - active.time) * fps);
  const clipStart = Math.max(0, active.time - 1.5);
  const clipEnd = Math.min(duration || active.time + 1.5, active.time + 1.5);
  const clipProgress = Math.max(0, Math.min(100, ((currentTime - clipStart) / Math.max(0.1, clipEnd - clipStart)) * 100));

  const confidenceColor = useMemo(() => active.confidence >= 0.85 ? "high" : active.confidence >= 0.72 ? "medium" : "low", [active.confidence]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><span>SL</span></div>
          <div><strong>ShuttleLab</strong><small>逐拍数据工作台</small></div>
        </div>

        <nav className="main-nav" aria-label="主导航">
          <button className="nav-item"><Icon>⌂</Icon><span>总览</span></button>
          <button className="nav-item"><Icon>▣</Icon><span>视频库</span><em>1</em></button>
          <button className="nav-item" onClick={() => setShowCalibration(true)}><Icon>⌗</Icon><span>球场标定</span>{manifest && <em>待确认</em>}</button>
          <button className="nav-item"><Icon>◈</Icon><span>处理任务</span><i className="status-dot" /></button>
          <button className="nav-item active"><Icon>⌁</Icon><span>事件标注</span><em>{events.filter((event) => event.status === "unreviewed").length}</em></button>
          <button className="nav-item"><Icon>◎</Icon><span>Gold 审核</span><em>{goldCount}</em></button>
          <button className="nav-item"><Icon>⌗</Icon><span>数据集</span></button>
          <button className="nav-item"><Icon>↗</Icon><span>模型实验</span></button>
        </nav>

        <div className="sidebar-section">
          <p>当前项目</p>
          <div className="project-card">
            <div className="project-thumb"><span>03</span></div>
            <div><strong>Fixed cam pilot</strong><small>{manifest?.video.file_name ?? "example.mp4"} · 单打</small></div>
          </div>
        </div>

        <div className="pipeline-mini">
          <div className="pipeline-head"><span>视频登记</span><b>{manifest ? "已验证" : "读取中"}</b></div>
          <div className="pipeline-bar"><span /></div>
          <small>{manifest ? `${manifest.video.frame_count.toLocaleString()} 帧 · SHA-256` : "读取真实视频 manifest"}</small>
        </div>

        <button className="profile">
          <span className="avatar">AN</span>
          <span><strong>Annotator 01</strong><small>本地工作站</small></span>
          <span className="profile-more">•••</span>
        </button>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <div className="breadcrumbs"><span>视频库</span><b>/</b><span>{manifest?.video.file_name ?? "example.mp4"}</span><b>/</b><strong>事件标注</strong></div>
            <div className="title-row">
              <h1>击球事件审核</h1>
              <span className="live-badge"><i /> {manifest ? "真实视频已登记" : "载入登记记录"}</span>
              {pilotMetrics && <div className="pilot-metrics" aria-label="Pilot Gold 评估指标">
                <span title="已确认的机器候选 ÷（已确认机器候选 + 已删除机器误报）">候选精度 <b>{pilotMetrics.candidate_precision === null ? "—" : `${Math.round(pilotMetrics.candidate_precision * 100)}%`}</b></span>
                <span title="估算值：已确认机器候选 ÷（已确认机器候选 + 人工补录 Gold）">估算召回 <b>{pilotMetrics.estimated_recall === null ? "—" : `${Math.round(pilotMetrics.estimated_recall * 100)}%`}</b><em>估</em></span>
                <span title="已完成独立复核与裁决的 Gold">双检 <b>{pilotMetrics.double_checked}</b> · 裁决 <b>{pilotMetrics.adjudicated}</b></span>
              </div>}
            </div>
          </div>
          <div className="top-actions">
            <span className={`sync-state ${syncState}`}><i />{
              syncState === "synced" ? "D1 已同步" : syncState === "saving" ? "正在保存" : syncState === "error" ? "同步异常" : "读取标注"
            }</span>
            <div className="review-progress">
              <span><b>{pilotMetrics?.gold ?? goldCount}</b> / {pilotMetrics?.target_gold ?? 50} Pilot Gold</span>
              <div><i style={{ width: `${(pilotMetrics?.progress ?? Math.min(1, goldCount / 50)) * 100}%` }} /></div>
            </div>
            <button className="ghost-button" onClick={() => setToast("快捷键：Space 播放 · ← → 逐帧 · Enter 保存")}><span>⌨</span> 快捷键</button>
            <button className="report-button" onClick={exportPilotReport}><span>◎</span> 评估报告</button>
            <button className="agreement-button" onClick={() => setShowAgreement(true)} disabled={!videoId}><span>≋</span> 一致性</button>
            <button className="export-button" onClick={exportCsv}><span>⇩</span> 导出 CSV</button>
          </div>
        </header>

        <div className="content-grid">
          <section className="event-rail">
            <div className="rail-head">
              <div><h2>候选事件</h2><span>{events.length} 条</span></div>
              <button className="rail-add" aria-label="在当前帧补录击球事件" onClick={addManualEvent}>＋ 补录</button>
            </div>
            <div className="queue-tabs">
              <button className={queueFilter === "all" ? "active" : ""} onClick={() => setQueueFilter("all")}>全部</button>
              <button className={queueFilter === "pending" ? "active" : ""} onClick={() => setQueueFilter("pending")}>待审核 <b>{events.filter((event) => event.status === "unreviewed").length}</b></button>
              <button className={queueFilter === "review" ? "active" : ""} onClick={() => setQueueFilter("review")}>待复核 <b>{reviewQueue.length}</b></button>
            </div>
            <div className="event-list">
              {visibleEvents.map((event, index) => (
                <button key={event.id} className={`event-row ${event.id === active.id ? "active" : ""}`} onClick={() => setActiveId(event.id)}>
                  <span className={`event-state ${event.status}`}>
                    {event.status === "gold" ? "✓" : event.status === "double_checked" ? "✓✓" : event.status === "adjudicated" ? "◆" : event.status === "rejected" ? "×" : event.status === "uncertain" ? "?" : String(index + 1).padStart(2, "0")}
                  </span>
                  <span className="event-copy">
                    <span className="event-meta"><strong>{event.id}</strong><small>{formatTime(event.time)}</small></span>
                    <span className="event-detail"><i className={`player-dot player-${event.hitter.toLowerCase()}`} />球员 {event.hitter}<b>·</b>{event.stroke}</span>
                  </span>
                  <span className={`confidence ${event.confidence >= 0.85 ? "high" : event.confidence >= 0.72 ? "medium" : "low"}`}>{Math.round(event.confidence * 100)}%</span>
                </button>
              ))}
            </div>
            <div className="rail-foot"><span className="kbd">N</span> 下一条待审核事件</div>
          </section>

          <section className="review-stage">
            <div className="stage-header">
              <div>
                <span className="rally-pill">{active.rally}</span>
                <span className="rally-actions">
                  <button title={draftDirty ? "请先保存当前标注" : "在当前事件前开始新回合"} disabled={!canSplitRally || draftDirty || syncState === "saving"} onClick={() => void persistRallyChanges(splitRallyBefore(events, active.id), "已在当前事件前拆分回合")}>拆分</button>
                  <button title={draftDirty ? "请先保存当前标注" : "把当前回合并入上一回合"} disabled={!canMergeRally || draftDirty || syncState === "saving"} onClick={() => void persistRallyChanges(mergeWithPreviousRally(events, active.rally), "已与上一回合合并")}>合并前回合</button>
                </span>
                <strong>{active.id}</strong>
                <small>候选帧 #{Math.round(active.time * fps)}</small>
                {active.revision && videoId && <button className="revision-chip" onClick={() => setHistoryEventId(active.id)}>标注 v{active.revision} · 查看历史</button>}
                {active.status === "double_checked" && <span className="review-status-chip checked">已双检</span>}
                {active.status === "adjudicated" && <span className="review-status-chip adjudicated">已裁决</span>}
                {manifest && <span className="manifest-chip" title={manifest.video.checksum}>{manifest.video.video_id}</span>}
              </div>
              <div className="overlay-toggles">
                <label className={showPose ? "on" : ""}><input type="checkbox" checked={showPose} onChange={(event) => setShowPose(event.target.checked)} />骨架</label>
                <label className={showTrail ? "on" : ""}><input type="checkbox" checked={showTrail} onChange={(event) => setShowTrail(event.target.checked)} />球轨迹</label>
                <button onClick={() => setToast("画面已适配当前窗口")}>⌗</button>
              </div>
            </div>

            <div className="video-wrap">
              <video
                ref={videoRef}
                src="/example.mp4"
                preload="metadata"
                playsInline
                onLoadedMetadata={(event) => {
                  const video = event.currentTarget;
                  setDuration(video.duration || 0);
                  video.currentTime = Math.min(active.time, Math.max(0, video.duration - 0.1));
                }}
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
              >
                <track kind="captions" src="/captions-empty.vtt" srcLang="zh-CN" label="源视频无音轨" default />
              </video>
              <div className="video-vignette" />
              {showPose && (
                <>
                  <div className="pose pose-upper"><i className="head" /><i className="body" /><i className="arm left" /><i className="arm right" /><i className="leg left" /><i className="leg right" /><b>A</b></div>
                  <div className="pose pose-lower"><i className="head" /><i className="body" /><i className="arm left" /><i className="arm right" /><i className="leg left" /><i className="leg right" /><b>B</b></div>
                </>
              )}
              {showTrail && <div className="shuttle-trail"><i /><i /><i /><i /><i /><b>●</b></div>}
              <button type="button" className={`spatial-overlay ${pointMode ? "active" : ""}`} disabled={!pointMode || !calibration} onPointerDown={placePointOnVideo} aria-label={pointMode ? `在视频中标记${pointMode === "hit" ? "击球点" : "落点"}` : "视频点位标记层"}>
                {manifest && hitPosition?.image_x !== null && hitPosition?.image_y !== null && hitPosition?.image_x !== undefined && hitPosition?.image_y !== undefined && <span className="spatial-marker hit" style={{ left: `${hitPosition.image_x / manifest.video.width * 100}%`, top: `${hitPosition.image_y / manifest.video.height * 100}%` }}>H</span>}
                {manifest && landingPosition?.image_x !== null && landingPosition?.image_y !== null && landingPosition?.image_x !== undefined && landingPosition?.image_y !== undefined && <span className="spatial-marker landing" style={{ left: `${landingPosition.image_x / manifest.video.width * 100}%`, top: `${landingPosition.image_y / manifest.video.height * 100}%` }}>L</span>}
              </button>
              <div className="frame-overlay"><span>FRAME</span><strong>{frame}</strong><small>{eventDelta === 0 ? "候选帧" : `${eventDelta > 0 ? "+" : ""}${eventDelta} 帧`}</small></div>
              <div className="video-status"><i /> {manifest ? `${manifest.video.width}×${manifest.video.height} · ${manifest.pipeline_version}` : "读取视频登记数据"}</div>
            </div>

            <div className="transport">
              <div className="timecode"><strong>{formatTime(currentTime)}</strong><span>/ {formatTime(duration)}</span></div>
              <div className="transport-buttons">
                <button onClick={() => stepFrames(-5)} aria-label="后退五帧"><span>−5</span></button>
                <button onClick={() => stepFrames(-1)} aria-label="后退一帧">‹<small>1</small></button>
                <button className="play-button" onClick={togglePlay} aria-label={playing ? "暂停" : "播放"}>{playing ? "Ⅱ" : "▶"}</button>
                <button onClick={() => stepFrames(1)} aria-label="前进一帧">›<small>1</small></button>
                <button onClick={() => stepFrames(5)} aria-label="前进五帧"><span>+5</span></button>
              </div>
              <div className="fps-tag">{fps} FPS</div>
            </div>

            <div className="timeline-card">
              <div className="timeline-labels"><span>{formatTime(clipStart)}</span><strong>3.0 秒击球窗口</strong><span>{formatTime(clipEnd)}</span></div>
              <div
                className="timeline-track"
                role="slider"
                tabIndex={0}
                aria-label="击球候选时间轴"
                aria-valuemin={clipStart}
                aria-valuemax={clipEnd}
                aria-valuenow={currentTime}
                onKeyDown={(event) => {
                  if (event.key === "ArrowLeft") stepFrames(-1);
                  if (event.key === "ArrowRight") stepFrames(1);
                }}
                onClick={(event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                seek(clipStart + ((event.clientX - rect.left) / rect.width) * (clipEnd - clipStart));
              }}>
                <div className="track-fill" style={{ width: `${clipProgress}%` }} />
                {[7, 16, 23, 31, 39, 47, 58, 66, 76, 86, 94].map((left, index) => <i key={left} style={{ left: `${left}%`, height: `${10 + (index % 4) * 4}px` }} />)}
                <div className="candidate-line" style={{ left: "50%" }}><b>候选</b></div>
                <div className="playhead" style={{ left: `${clipProgress}%` }} />
              </div>
              <div className="frame-hint"><span><kbd>←</kbd><kbd>→</kbd> 前后 1 帧</span><span><kbd>Shift</kbd> + <kbd>←</kbd><kbd>→</kbd> 前后 5 帧</span></div>
            </div>

            <div className="context-strip">
              <button onClick={() => activeIndex > 0 && setActiveId(events[activeIndex - 1].id)} disabled={activeIndex === 0}><span>上一拍</span><strong>{activeIndex > 0 ? events[activeIndex - 1].stroke : "—"}</strong></button>
              <div><i className="player-dot player-a" /><span>当前回合</span><strong>第 {activeIndex + 1} 拍</strong></div>
              <button onClick={() => activeIndex < events.length - 1 && setActiveId(events[activeIndex + 1].id)} disabled={activeIndex === events.length - 1}><span>下一拍</span><strong>{activeIndex < events.length - 1 ? events[activeIndex + 1].stroke : "—"}</strong></button>
            </div>
          </section>

          <aside className="annotation-panel">
            <div className="panel-scroll">
              <div className="prediction-card">
                <div className="prediction-top"><span>机器预测</span><b className={confidenceColor}>{Math.round(active.confidence * 100)}% 置信度</b></div>
                <div className="prediction-main"><div className={`player-token player-${active.hitter.toLowerCase()}`}>{active.hitter}</div><div><strong>球员 {active.hitter} · {active.stroke}</strong><small>Hit rules v0.2</small></div></div>
                <div className="signal-list">
                  <div><span>轨迹突变</span><i><b style={{ width: `${active.trajectory * 100}%` }} /></i><em>{Math.round(active.trajectory * 100)}</em></div>
                  <div><span>手腕距离</span><i><b style={{ width: `${active.wrist * 100}%` }} /></i><em>{Math.round(active.wrist * 100)}</em></div>
                  <div><span>姿态幅度</span><i><b style={{ width: `${active.pose * 100}%` }} /></i><em>{Math.round(active.pose * 100)}</em></div>
                </div>
              </div>

              <section className="form-section">
                <div className="form-title"><span>01</span><div><h3>击球者</h3><small>谁完成了这次击球？</small></div></div>
                <div className="hitter-options">
                  <button className={hitter === "A" ? "selected a" : ""} onClick={() => setHitter("A")}><i>A</i><span><strong>球员 A</strong><small>{currentSide.upper_player === "A" ? "画面上半场" : "画面下半场"}</small></span><kbd>A</kbd></button>
                  <button className={hitter === "B" ? "selected b" : ""} onClick={() => setHitter("B")}><i>B</i><span><strong>球员 B</strong><small>{currentSide.upper_player === "B" ? "画面上半场" : "画面下半场"}</small></span><kbd>B</kbd></button>
                </div>
              </section>

              <section className="form-section">
                <div className="form-title"><span>02</span><div><h3>技术动作</h3><small>选择粗粒度动作类型</small></div></div>
                <div className="stroke-grid">
                  {STROKES.map((item, index) => <button key={item} className={stroke === item ? "selected" : ""} onClick={() => setStroke(item)}><kbd>{index + 1}</kbd>{item}{stroke === item && <i>✓</i>}</button>)}
                </div>
              </section>

              <section className="form-section compact">
                <div className="form-title"><span>03</span><div><h3>补充属性</h3><small>可选</small></div></div>
                <div className="property-row"><span><strong>反手</strong><small>Backhand</small></span><button className={`switch ${backhand ? "on" : ""}`} onClick={() => setBackhand(!backhand)} aria-label="切换反手"><i /></button></div>
                <div className="property-row"><span><strong>头顶区</strong><small>Around head</small></span><button className={`switch ${aroundhead ? "on" : ""}`} onClick={() => setAroundhead(!aroundhead)} aria-label="切换头顶区"><i /></button></div>
              </section>

              <section className="form-section compact">
                <div className="form-title"><span>04</span><div><h3>标注质量</h3><small>你的判断把握</small></div></div>
                <div className="certainty-options">{["确定", "基本确定", "无法判断"].map((item) => <button key={item} className={certainty === item ? "selected" : ""} onClick={() => setCertainty(item)}>{item}</button>)}</div>
                <textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="备注（可选）：遮挡、边界情况或复核说明…" />
              </section>

              <section className="form-section compact">
                <div className="form-title"><span>05</span><div><h3>空间点位</h3><small>在视频或俯视球场上点击修正</small></div></div>
                <div className="point-tools">
                  <button className={pointMode === "hit" ? "selected hit" : ""} onClick={() => setPointMode(pointMode === "hit" ? null : "hit")}><kbd>H</kbd><span><strong>击球点</strong><small>{hitPosition ? `${hitPosition.court_x.toFixed(2)}, ${hitPosition.court_y.toFixed(2)}m` : "未标记"}</small></span></button>
                  <button className={pointMode === "landing" ? "selected landing" : ""} onClick={() => setPointMode(pointMode === "landing" ? null : "landing")}><kbd>L</kbd><span><strong>落点</strong><small>{landingPosition ? `${landingPosition.court_x.toFixed(2)}, ${landingPosition.court_y.toFixed(2)}m` : "未标记"}</small></span></button>
                </div>
                <div className="point-help"><span>{pointMode ? `正在标记${pointMode === "hit" ? "击球点" : "落点"} · 点击视频或下方球场` : "选择点位类型后开始"}</span>{(hitPosition || landingPosition) && <button onClick={() => { setHitPosition(null); setLandingPosition(null); setPointMode(null); }}>清除点位</button>}</div>
                {!calibration && <p className="point-warning">视频点击需要先完成球场标定；仍可在下方俯视球场直接标记。</p>}
              </section>

              <section className="form-section compact review-workflow">
                <div className="form-title"><span>06</span><div><h3>Gold 复核</h3><small>独立双检与分歧裁决均追加历史</small></div></div>
                <div className="review-flow">
                  <span className={["gold", "uncertain"].includes(active.status) ? "current" : ["double_checked", "adjudicated"].includes(active.status) ? "done" : ""}>已标注</span><i>→</i>
                  <span className={active.status === "double_checked" ? "current" : active.status === "adjudicated" ? "done" : ""}>已双检</span><i>→</i>
                  <span className={active.status === "adjudicated" ? "current" : ""}>已裁决</span>
                </div>
                <textarea className="review-note" value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} placeholder="复核意见或分歧说明（裁决不确定项时必填）" disabled={active.status === "unreviewed" || active.status === "rejected" || active.status === "adjudicated"} />
                <div className="review-buttons">
                  <button onClick={() => void saveStatus("double_checked")} disabled={syncState === "saving" || !["gold", "uncertain"].includes(active.status)}>✓ 复核通过</button>
                  <button onClick={() => void saveStatus("adjudicated")} disabled={syncState === "saving" || !["double_checked", "uncertain"].includes(active.status) || (active.status === "uncertain" && !reviewNote.trim())}>◆ 裁决确认</button>
                </div>
                {active.status === "unreviewed" && <p>先完成基础标注，事件进入 Gold 后即可独立复核。</p>}
                {active.status === "adjudicated" && <p>该事件已完成裁决；后续更改仍会保留在修订历史中。</p>}
              </section>

              <section className="court-card">
                <div className="court-head">
                  <span><strong>球场位置</strong><small>场侧映射 v{currentSide.version} · 当前帧生效</small></span>
                  <button className="side-switch-button" disabled={draftDirty || syncState === "saving"} title={draftDirty ? "请先保存当前标注" : "从当前帧交换 A/B 场侧"} onClick={() => void recordSideSwitch()}>⇅ 记录换边</button>
                </div>
                <div className={`court ${pointMode ? "placing-point" : ""}`} onPointerDown={placePointOnCourt}>
                  <i className="court-mid" /><i className="service-line top" /><i className="service-line bottom" /><i className="center-line top" /><i className="center-line bottom" />
                  <span className={`court-player upper player-${currentSide.upper_player.toLowerCase()}`}>{currentSide.upper_player}</span><span className={`court-player lower player-${currentSide.lower_player.toLowerCase()}`}>{currentSide.lower_player}</span><span className="court-shuttle">•</span>
                  {hitPosition && <span className="court-spatial-marker hit" style={{ left: `${hitPosition.court_x / 6.1 * 100}%`, top: `${hitPosition.court_y / 13.4 * 100}%` }}>H</span>}
                  {landingPosition && <span className="court-spatial-marker landing" style={{ left: `${landingPosition.court_x / 6.1 * 100}%`, top: `${landingPosition.court_y / 13.4 * 100}%` }}>L</span>}
                </div>
                <div className="court-coords"><span>远场 <b>{currentSide.upper_player}</b></span><span>近场 <b>{currentSide.lower_player}</b></span></div>
                {sideAssignments.length > 0 && <div className="side-history">
                  {sideAssignments.slice(-3).reverse().map((assignment) => <span key={assignment.id}>#{assignment.effective_frame} · 远场 {assignment.upper_player} / 近场 {assignment.lower_player}</span>)}
                </div>}
              </section>
            </div>

            <div className="panel-actions">
              <div className="secondary-actions">
                <button onClick={() => void saveStatus(active.status === "rejected" ? "unreviewed" : "rejected")} disabled={syncState === "saving" || isReviewLocked}><kbd>{active.status === "rejected" ? "R" : "X"}</kbd> {active.status === "rejected" ? "恢复候选" : "删除误报"}</button>
                <button onClick={() => void saveStatus("uncertain")} disabled={syncState === "saving" || isReviewLocked}><kbd>U</kbd> 送去复核</button>
              </div>
              <button className="save-button" disabled={!videoId || syncState === "saving" || isReviewLocked} onClick={() => void saveStatus(certainty === "无法判断" ? "uncertain" : "gold")}><span><strong>{syncState === "saving" ? "正在保存…" : isReviewLocked ? "通过复核区继续" : "保存并下一条"}</strong><small>D1 追加版本，不覆盖历史</small></span><kbd>Enter ↵</kbd></button>
            </div>
          </aside>
        </div>
      </section>
      {showCalibration && manifest && <CalibrationModal video={manifest.video} onClose={() => setShowCalibration(false)} onSaved={(message, savedCalibration) => {
        setToast(message);
        if (savedCalibration) setCalibration(savedCalibration);
      }} />}
      {historyEventId && videoId && <EventHistoryModal videoId={videoId} eventId={historyEventId} onClose={() => setHistoryEventId(null)} />}
      {showAgreement && videoId && <AgreementModal videoId={videoId} onClose={() => setShowAgreement(false)} />}
      {toast && <div className="toast"><i>✓</i>{toast}</div>}
    </main>
  );
}
