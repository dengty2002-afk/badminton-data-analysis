"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { computeHomography, validateCalibration, type CalibrationPoint } from "../lib/homography";

type CalibrationVideo = {
  video_id: string;
  file_name: string;
  width: number;
  height: number;
  duration_sec: number;
};

type SavedCalibration = {
  id: string;
  video_id: string;
  version: number;
  frame_time_ms: number;
  image_width: number;
  image_height: number;
  corners: CalibrationPoint[];
  homography: number[][];
  quality_status: string;
  source: string;
  created_at: string;
};

const POINT_LABELS = ["上方左角", "上方右角", "下方右角", "下方左角"];

function formatTime(seconds: number) {
  const min = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function CalibrationEdge({ start, end }: { start: CalibrationPoint; end: CalibrationPoint }) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const width = Math.hypot(dx, dy * 0.5625);
  const angle = Math.atan2(dy * 0.5625, dx) * 180 / Math.PI;
  return <i className="calibration-edge" style={{ left: `${start.x * 100}%`, top: `${start.y * 100}%`, width: `${width * 100}%`, transform: `rotate(${angle}deg)` }} />;
}

export function CalibrationModal({ video, onClose, onSaved }: { video: CalibrationVideo; onClose: () => void; onSaved: (message: string, calibration?: SavedCalibration) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [points, setPoints] = useState<CalibrationPoint[]>([]);
  const [dragging, setDragging] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(video.duration_sec || 0);
  const [playing, setPlaying] = useState(false);
  const [saved, setSaved] = useState<SavedCalibration[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const pixelPoints = useMemo(() => points.map((point) => ({ x: point.x * video.width, y: point.y * video.height })), [points, video.height, video.width]);
  const validation = useMemo(() => validateCalibration(pixelPoints, video.width, video.height), [pixelPoints, video.height, video.width]);
  const matrix = useMemo(() => {
    if (!validation.valid) return null;
    try { return computeHomography(pixelPoints); } catch { return null; }
  }, [pixelPoints, validation.valid]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/calibrations?video_id=${encodeURIComponent(video.video_id)}`, { signal: controller.signal })
      .then(async (response) => {
        const body = await response.json() as { calibrations?: SavedCalibration[]; error?: string };
        if (!response.ok) throw new Error(body.error ?? "无法读取标定历史");
        setSaved(body.calibrations ?? []);
      })
      .catch((reason) => {
        if (reason instanceof Error && reason.name !== "AbortError") setError(reason.message);
      })
      .finally(() => setLoadingHistory(false));
    return () => controller.abort();
  }, [video.video_id]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key.toLowerCase() === "r") setPoints([]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const pointFromEvent = (clientX: number, clientY: number) => {
    const rect = overlayRef.current?.getBoundingClientRect();
    if (!rect) return null;
    return {
      x: Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (clientY - rect.top) / rect.height)),
    };
  };

  const saveCalibration = async () => {
    if (!matrix || !validation.valid) return;
    setSaving(true);
    setError("");
    try {
      const response = await fetch("/api/calibrations", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          video_id: video.video_id,
          frame_time_ms: Math.round(currentTime * 1000),
          image_width: video.width,
          image_height: video.height,
          corners: pixelPoints,
        }),
      });
      const body = await response.json() as { calibration?: SavedCalibration; error?: string };
      if (!response.ok || !body.calibration) throw new Error(body.error ?? "保存失败");
      setSaved((current) => [body.calibration!, ...current]);
      onSaved(`球场标定 v${body.calibration.version} 已保存`, body.calibration);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "保存失败");
    } finally {
      setSaving(false);
    }
  };

  const downloadJson = () => {
    const record = saved[0] ?? (matrix ? {
      video_id: video.video_id,
      version: 0,
      frame_time_ms: Math.round(currentTime * 1000),
      image_width: video.width,
      image_height: video.height,
      corners: pixelPoints,
      homography: matrix,
      quality_status: validation.valid ? "draft_valid" : "draft_invalid",
      source: "manual",
      created_at: new Date().toISOString(),
    } : null);
    if (!record) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${video.video_id}_calibration_v${record.version}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="calibration-modal" role="dialog" aria-modal="true" aria-label="球场四角标定">
      <header className="calibration-topbar">
        <div className="calibration-title">
          <span className="calibration-step">PIPELINE 02</span>
          <div><h2>球场四角标定</h2><p>{video.file_name} · {video.video_id}</p></div>
        </div>
        <div className="calibration-top-actions">
          <span className="calibration-save-state"><i /> D1 持久化</span>
          <button onClick={onClose}>返回标注台 <kbd>Esc</kbd></button>
        </div>
      </header>

      <div className="calibration-body">
        <section className="calibration-main">
          <div className="calibration-guide">
            <div><strong>在球场外侧边线角点上依次点击</strong><span>按从上到下的顺时针顺序，点击后可拖动微调</span></div>
            <div className="corner-order">
              {POINT_LABELS.map((label, index) => <span key={label} className={points.length > index ? "done" : points.length === index ? "current" : ""}><b>{points.length > index ? "✓" : index + 1}</b>{label}</span>)}
            </div>
          </div>

          <div className="calibration-video-shell">
            <video
              ref={videoRef}
              src="/example.mp4"
              preload="metadata"
              playsInline
              onLoadedMetadata={(event) => {
                setDuration(event.currentTarget.duration || video.duration_sec);
                event.currentTarget.currentTime = Math.min(2, Math.max(0, event.currentTarget.duration - 0.1));
              }}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
            >
              <track kind="captions" src="/captions-empty.vtt" srcLang="zh-CN" label="源视频无音轨" default />
            </video>
            <div
              ref={overlayRef}
              className={`calibration-overlay ${points.length < 4 ? "placing" : "complete"}`}
              onPointerDown={(event) => {
                if (points.length >= 4 || dragging !== null) return;
                const point = pointFromEvent(event.clientX, event.clientY);
                if (point) setPoints((current) => [...current, point]);
              }}
              onPointerMove={(event) => {
                if (dragging === null) return;
                const point = pointFromEvent(event.clientX, event.clientY);
                if (point) setPoints((current) => current.map((item, index) => index === dragging ? point : item));
              }}
              onPointerUp={() => setDragging(null)}
              onPointerLeave={() => setDragging(null)}
            >
              {points.map((point, index) => {
                const next = points[(index + 1) % points.length];
                const shouldDraw = index < points.length - 1 || points.length === 4;
                return <span key={index}>{shouldDraw && next && <CalibrationEdge start={point} end={next} />}<button className="calibration-point" style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }} onPointerDown={(event) => { event.stopPropagation(); setDragging(index); }} aria-label={`调整${POINT_LABELS[index]}`}><b>{index + 1}</b><em>{POINT_LABELS[index]}</em></button></span>;
              })}
              {points.length < 4 && <div className="calibration-cursor-hint">点击：{POINT_LABELS[points.length]}</div>}
            </div>
            <div className="calibration-frame-tag">标定帧 <b>{Math.round(currentTime * 1000)} ms</b></div>
          </div>

          <div className="calibration-transport">
            <button className="calibration-play" onClick={() => {
              const element = videoRef.current;
              if (!element) return;
              if (element.paused) element.play().catch(() => undefined); else element.pause();
            }}>{playing ? "Ⅱ" : "▶"}</button>
            <span>{formatTime(currentTime)}</span>
            <input type="range" min={0} max={duration || 1} step={0.033} value={currentTime} onChange={(event) => {
              const time = Number(event.target.value);
              if (videoRef.current) { videoRef.current.pause(); videoRef.current.currentTime = time; }
              setCurrentTime(time);
            }} />
            <span>{formatTime(duration)}</span>
            <button className="calibration-reset" onClick={() => setPoints([])}>重置角点 <kbd>R</kbd></button>
          </div>
        </section>

        <aside className="calibration-panel">
          <section className="calibration-status-card">
            <div className="calibration-section-title"><span>01</span><div><h3>几何质量</h3><small>保存前自动检查</small></div></div>
            <div className={`geometry-result ${validation.valid ? "valid" : "pending"}`}><i>{validation.valid ? "✓" : points.length}</i><div><strong>{validation.valid ? "检查通过" : "等待角点"}</strong><span>{validation.message}</span></div></div>
            <div className="geometry-metric"><span>画面覆盖率</span><strong>{(validation.areaRatio * 100).toFixed(2)}%</strong></div>
            <div className="geometry-metric"><span>球场尺寸</span><strong>6.10 × 13.40 m</strong></div>
            <div className="geometry-metric"><span>方向</span><strong>上方为远端</strong></div>
          </section>

          <section className="homography-card">
            <div className="calibration-section-title"><span>02</span><div><h3>Homography</h3><small>图像坐标 → 标准球场米制坐标</small></div></div>
            <div className="matrix-grid">
              {(matrix ?? [[0,0,0],[0,0,0],[0,0,1]]).flat().map((value, index) => <code key={index}>{value.toExponential(3)}</code>)}
            </div>
          </section>

          <section className="court-preview-card">
            <div className="calibration-section-title"><span>03</span><div><h3>俯视预览</h3><small>标准单打球场</small></div></div>
            <div className="calibration-court"><i className="net" /><i className="service top" /><i className="service bottom" /><i className="center top" /><i className="center bottom" /><b className="corner c1">1</b><b className="corner c2">2</b><b className="corner c3">3</b><b className="corner c4">4</b></div>
          </section>

          <section className="calibration-history">
            <div className="history-title"><strong>版本历史</strong><span>{loadingHistory ? "读取中" : `${saved.length} 个版本`}</span></div>
            {saved.slice(0, 3).map((record) => <div className="history-row" key={record.id}><i>v{record.version}</i><span><strong>{record.source === "manual" ? "人工标定" : record.source}</strong><small>{new Date(record.created_at).toLocaleString("zh-CN")}</small></span><b>{record.quality_status}</b></div>)}
            {!loadingHistory && saved.length === 0 && <p>尚无已保存版本。完成四角选择后保存 v1。</p>}
          </section>

          {error && <div className="calibration-error">{error}</div>}
          <div className="calibration-actions">
            <button className="download-calibration" onClick={downloadJson} disabled={!matrix && saved.length === 0}>⇩ 下载 JSON</button>
            <button className="save-calibration" onClick={saveCalibration} disabled={!validation.valid || saving}>{saving ? "保存中…" : `保存标定 v${(saved[0]?.version ?? 0) + 1}`}</button>
          </div>
        </aside>
      </div>
    </div>
  );
}
