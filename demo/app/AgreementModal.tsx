"use client";

import { useEffect, useState } from "react";

type Disagreement = {
  event_id: string;
  base_revision: number;
  review_revision: number;
  changed_fields: string[];
  original: { hitter: string; hit_frame: number; stroke: string };
  review: { hitter: string; hit_frame: number; stroke: string };
  reviewer_id: string | null;
  review_note: string | null;
};

type AgreementReport = {
  video_id: string;
  generated_at: string;
  sample_size: number;
  adjudicated_count: number;
  hitter_agreement: number | null;
  hit_frame_within_3_agreement: number | null;
  stroke_agreement: number | null;
  stroke_cohen_kappa: number | null;
  disagreements_count: number;
  targets: { hitter_agreement: number; hit_frame_within_3_agreement: number; stroke_cohen_kappa: number };
  disagreements: Disagreement[];
};

function percent(value: number | null) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function Metric({ label, value, target }: { label: string; value: number | null; target: number }) {
  const passed = value != null && value >= target;
  return <div className={`agreement-metric ${value == null ? "empty" : passed ? "passed" : "below"}`}>
    <span>{label}</span><strong>{percent(value)}</strong><small>目标 ≥ {percent(target)}</small>
  </div>;
}

export function AgreementModal({ videoId, onClose }: { videoId: string; onClose: () => void }) {
  const [report, setReport] = useState<AgreementReport | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/events/agreement?video_id=${encodeURIComponent(videoId)}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json() as { report?: AgreementReport; error?: string };
        if (!response.ok || !payload.report) throw new Error(payload.error || "读取一致性报告失败");
        setReport(payload.report);
      })
      .catch((reason) => {
        if ((reason as Error).name !== "AbortError") setError(reason instanceof Error ? reason.message : "读取一致性报告失败");
      });
    return () => controller.abort();
  }, [videoId]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const download = () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `shuttlelab_agreement_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return <div className="history-backdrop" onPointerDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="history-modal agreement-modal" role="dialog" aria-modal="true" aria-label="Gold 标注一致性报告">
      <header>
        <div><span>GOLD AGREEMENT</span><h2>标注一致性报告</h2><p>基于初始标注与独立双检修订自动计算。</p></div>
        <div className="agreement-head-actions"><button onClick={download} disabled={!report}>⇩ JSON</button><button onClick={onClose} aria-label="关闭一致性报告">×</button></div>
      </header>
      {!report && !error && <p className="history-empty">正在计算 D1 双检样本…</p>}
      {error && <p className="history-error">{error}</p>}
      {report && <div className="agreement-body">
        <div className="agreement-summary"><span>双检样本 <b>{report.sample_size}</b></span><span>分歧 <b>{report.disagreements_count}</b></span><span>已裁决 <b>{report.adjudicated_count}</b></span></div>
        <div className="agreement-metrics">
          <Metric label="击球者一致率" value={report.hitter_agreement} target={report.targets.hitter_agreement} />
          <Metric label="击球帧 ±3" value={report.hit_frame_within_3_agreement} target={report.targets.hit_frame_within_3_agreement} />
          <Metric label="动作一致率" value={report.stroke_agreement} target={0.8} />
          <Metric label="动作 Cohen's κ" value={report.stroke_cohen_kappa} target={report.targets.stroke_cohen_kappa} />
        </div>
        {report.sample_size === 0 && <p className="agreement-empty">完成第一条“复核通过”后，这里会自动出现一致性指标。</p>}
        {report.disagreements.length > 0 && <section className="disagreement-section">
          <div className="history-title"><strong>分歧清单</strong><span>{report.disagreements.length} 条</span></div>
          {report.disagreements.map((item) => <article key={`${item.event_id}-${item.review_revision}`} className="disagreement-row">
            <div><strong>{item.event_id}</strong><span>v{item.base_revision} → v{item.review_revision}</span></div>
            <p><b>初标</b> {item.original.hitter} · #{item.original.hit_frame} · {item.original.stroke}</p>
            <p><b>复核</b> {item.review.hitter} · #{item.review.hit_frame} · {item.review.stroke}</p>
            <small>差异：{item.changed_fields.join(" / ")}{item.review_note ? ` · ${item.review_note}` : ""}</small>
          </article>)}
        </section>}
      </div>}
    </section>
  </div>;
}
