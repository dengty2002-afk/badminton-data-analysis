"use client";

import { useEffect, useState } from "react";

type Revision = {
  id: string;
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
  hit_position: { court_x: number; court_y: number } | null;
  landing_position: { court_x: number; court_y: number } | null;
  status: "unreviewed" | "gold" | "rejected" | "uncertain" | "double_checked" | "adjudicated";
  label_source: "machine_review" | "manual";
  machine_prediction: Record<string, unknown>;
  annotator_id: string;
  reviewer_id: string | null;
  review_note: string | null;
  review_base_revision: number | null;
  reviewed_at: string | null;
  created_at: string;
};

const STATUS_LABELS: Record<Revision["status"], string> = {
  unreviewed: "待审核",
  gold: "Gold",
  rejected: "误报",
  uncertain: "待复核",
  double_checked: "已双检",
  adjudicated: "已裁决",
};

export function EventHistoryModal({ videoId, eventId, onClose }: { videoId: string; eventId: string; onClose: () => void }) {
  const [revisions, setRevisions] = useState<Revision[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/events?video_id=${encodeURIComponent(videoId)}&event_id=${encodeURIComponent(eventId)}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json() as { revisions?: Revision[]; error?: string };
        if (!response.ok || !payload.revisions) throw new Error(payload.error || "读取修订历史失败");
        setRevisions(payload.revisions);
      })
      .catch((reason) => {
        if ((reason as Error).name !== "AbortError") setError(reason instanceof Error ? reason.message : "读取修订历史失败");
      });
    return () => controller.abort();
  }, [eventId, videoId]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div className="history-backdrop" onPointerDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="history-modal" role="dialog" aria-modal="true" aria-label={`${eventId} 修订历史`}>
        <header>
          <div><span>GOLD AUDIT</span><h2>{eventId} 修订历史</h2><p>每次保存均追加新版本，历史记录不会被覆盖。</p></div>
          <button onClick={onClose} aria-label="关闭修订历史">×</button>
        </header>
        <div className="history-list" aria-live="polite">
          {!error && revisions.length === 0 && <p className="history-empty">正在读取 D1 历史…</p>}
          {error && <p className="history-error">{error}</p>}
          {revisions.map((revision) => {
            const predictedHitter = String(revision.machine_prediction.hitter ?? "—");
            const predictedStroke = String(revision.machine_prediction.stroke ?? "未记录");
            return <article key={revision.id} className={`history-revision ${revision.status}`}>
              <div className="history-revision-head">
                <span>v{revision.revision}</span>
                <strong>{STATUS_LABELS[revision.status]}</strong>
                <time dateTime={revision.created_at}>{new Date(revision.created_at).toLocaleString("zh-CN", { hour12: false })}</time>
              </div>
              <div className="history-compare">
                <div><small>机器原预测</small><b>{predictedHitter} · {predictedStroke}</b></div>
                <i>→</i>
                <div><small>人工修订</small><b>{revision.hitter} · {revision.stroke}</b></div>
              </div>
              <dl>
                <div><dt>回合</dt><dd>{revision.rally_id}</dd></div>
                <div><dt>击球位置</dt><dd>#{revision.candidate_frame} · {revision.candidate_time.toFixed(3)}s</dd></div>
                <div><dt>击球点</dt><dd>{revision.hit_position ? `(${revision.hit_position.court_x.toFixed(2)}, ${revision.hit_position.court_y.toFixed(2)}m)` : "未标记"}</dd></div>
                <div><dt>落点</dt><dd>{revision.landing_position ? `(${revision.landing_position.court_x.toFixed(2)}, ${revision.landing_position.court_y.toFixed(2)}m)` : "未标记"}</dd></div>
                <div><dt>附加属性</dt><dd>{revision.backhand ? "反手" : "正手"}{revision.aroundhead ? " · 头顶区" : ""}</dd></div>
                <div><dt>标注质量</dt><dd>{revision.certainty}</dd></div>
                <div><dt>来源</dt><dd>{revision.label_source === "manual" ? "人工补录" : "机器候选审核"}</dd></div>
                <div><dt>标注者</dt><dd>{revision.annotator_id}</dd></div>
                {revision.reviewer_id && <div><dt>{revision.status === "adjudicated" ? "裁决者" : "复核者"}</dt><dd>{revision.reviewer_id}</dd></div>}
                {revision.review_base_revision && <div><dt>复核基线</dt><dd>v{revision.review_base_revision}</dd></div>}
              </dl>
              {revision.notes && <p className="history-notes">备注：{revision.notes}</p>}
              {revision.review_note && <p className="history-notes">复核意见：{revision.review_note}</p>}
            </article>;
          })}
        </div>
      </section>
    </div>
  );
}
