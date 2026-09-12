/* eslint-disable react-hooks/set-state-in-effect, jsx-a11y/media-has-caption -- This video annotation workspace resets editor state when its external video/anchor selection changes; source match videos contain no dialogue track. */
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Anchor = { anchorId: string; rallyId: string; hitFrame: number; hitter: "upper" | "lower" | "unknown"; confidence: string; uncertaintyFrames: number; notes: string };
type GoldVideo = { videoId: string; fileName: string; fps: number; frameCount: number; durationSec: number; anchorCount: number; anchors: Anchor[] };
type Catalog = { schemaVersion: string; videoCount: number; anchorCount: number; videos: GoldVideo[] };
type Annotation = {
  id?: string; video_id: string; anchor_id: string; revision?: number; rally_id: string; hit_frame: number;
  hitter: Anchor["hitter"]; stroke_type: string; landing_kind: "next_contact" | "terminal";
  landing_status: "observed" | "out" | "caught" | "unobservable"; landing_area: number | null;
  landing_area_schema?: "shuttlelab_9x7_v1" | "shuttlelab_9_v2";
  confidence: "high" | "medium" | "low"; uncertainty_frames: number; notes: string;
  status: "gold" | "uncertain";
};

const STROKES = [
  ["net shot", "放网", "网前轻放，使球贴网下坠"], ["return net", "回放网", "对方网前球后的再次贴网回放"],
  ["smash", "杀球", "高点快速向下进攻"], ["wrist smash", "点杀", "主要依靠手腕发力的短促下压"],
  ["lob", "挑球", "从前场向对方后场挑高"], ["defensive return lob", "防守挑球", "受压防守时向后场挑高"],
  ["clear", "高远球", "从后场打向对方后场的高弧线球"], ["drive", "平抽", "中前场快速、较平的对抽"],
  ["driven flight", "平高球", "比高远球更平、更快地压向后场"], ["back-court drive", "后场平抽", "从后场发出的平快抽击"],
  ["drop", "吊球", "后场主动减速落向前场"], ["passive drop", "被动吊球", "受压状态下从后场过渡到前场"],
  ["push", "推球", "网前或中前场向后推送"], ["rush", "扑球", "网前抢高点击球下压"],
  ["defensive return drive", "接杀挡抽", "防守杀球后的平快回击"], ["cross-court net shot", "勾对角", "网前斜线越网到对角"],
  ["short service", "短发球", "发球落向前发球线附近"], ["long service", "长发球", "发球送向后场"],
] as const;

const COURT_AREAS = [
  [1, "前左"], [2, "前中"], [3, "前右"],
  [4, "中左"], [5, "中路"], [6, "中右"],
  [7, "后左"], [8, "后中"], [9, "后右"],
] as const;
function isCompleteGold(annotation?: Annotation) {
  return annotation?.status === "gold" && Boolean(annotation.stroke_type) &&
    (annotation.landing_status === "out" || annotation.landing_status === "caught" || annotation.landing_status === "unobservable" ||
      (annotation.landing_status === "observed" && annotation.landing_area != null));
}

function csvCell(value: unknown) {
  const text = value == null ? "" : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function download(name: string, body: string, type = "text/csv;charset=utf-8") {
  const url = URL.createObjectURL(new Blob(["\uFEFF", body], { type }));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url);
}

const DIRECTORY_INPUT_PROPS = { webkitdirectory: "", directory: "" } as Record<string, string>;

export default function SemanticGoldPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [videoIndex, setVideoIndex] = useState(0);
  const [anchorIndex, setAnchorIndex] = useState(0);
  const [annotations, setAnnotations] = useState<Record<string, Annotation>>({});
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoNeedsFile, setVideoNeedsFile] = useState(false);
  const [localVideoUrls, setLocalVideoUrls] = useState<Record<string, string>>({});
  const [sourceStatus, setSourceStatus] = useState("尚未选择本机素材文件夹");
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [strokeType, setStrokeType] = useState("");
  const [landingStatus, setLandingStatus] = useState<Annotation["landing_status"]>("observed");
  const [landingArea, setLandingArea] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<"high" | "medium" | "low">("high");
  const [uncertainty, setUncertainty] = useState(1);
  const [notes, setNotes] = useState("");
  const [sync, setSync] = useState("正在载入 Gold 目录…");
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/gold/catalog.json").then((response) => {
      if (!response.ok) throw new Error("Gold 目录不可用"); return response.json() as Promise<Catalog>;
    }).then((value) => { setCatalog(value); setSync(`已载入 ${value.anchorCount} 条击球锚点`); })
      .catch((reason: Error) => { setError(reason.message); setSync("载入失败"); });
  }, []);

  const activeVideo = catalog?.videos[videoIndex] ?? null;
  const activeAnchor = activeVideo?.anchors[anchorIndex] ?? null;
  const nextAnchor = activeVideo?.anchors[anchorIndex + 1] ?? null;
  const sameRallyNext = Boolean(activeAnchor && nextAnchor && activeAnchor.rallyId === nextAnchor.rallyId);
  const connectedVideoCount = Object.keys(localVideoUrls).length;

  const registerFiles = useCallback((files: Iterable<File>, sourceLabel: string) => {
    if (!catalog) return 0;
    const wanted = new Set(catalog.videos.map((video) => video.fileName));
    const matches = [...files].filter((file) => wanted.has(file.name));
    setLocalVideoUrls((current) => {
      const next = { ...current };
      for (const file of matches) next[file.name] = URL.createObjectURL(file);
      return next;
    });
    setSourceStatus(matches.length ? `${sourceLabel} · 本次匹配 ${matches.length} 段` : `${sourceLabel}中没有登记视频`);
    return matches.length;
  }, [catalog]);

  const seekFrame = useCallback((frame: number) => {
    if (!activeVideo) return;
    const clamped = Math.max(0, Math.min(activeVideo.frameCount - 1, Math.round(frame)));
    if (videoRef.current) { videoRef.current.pause(); videoRef.current.currentTime = clamped / activeVideo.fps; }
    setPlaying(false); setCurrentFrame(clamped);
  }, [activeVideo]);

  useEffect(() => {
    if (!activeVideo) return;
    setAnnotations({}); setAnchorIndex(0); setVideoNeedsFile(false); setError("");
    setSync("正在读取已保存标注…");
    fetch(`/api/semantic-gold?video_id=${encodeURIComponent(activeVideo.videoId)}`)
      .then(async (response) => { const data = await response.json() as { annotations?: Annotation[]; error?: string }; if (!response.ok) throw new Error(data.error || "读取失败"); return data; })
      .then((data) => {
        const map = Object.fromEntries((data.annotations ?? []).map((row) => [row.anchor_id, row]));
        setAnnotations(map);
        const firstIncomplete = activeVideo.anchors.findIndex((anchor) => !isCompleteGold(map[anchor.anchorId]));
        setAnchorIndex(firstIncomplete < 0 ? 0 : firstIncomplete);
        setSync(`数据库已同步 · ${Object.keys(map).length}/${activeVideo.anchorCount}`);
      }).catch((reason: Error) => { setError(reason.message); setSync("数据库暂不可用"); });
  }, [activeVideo]);

  useEffect(() => {
    if (!activeVideo) return;
    setVideoNeedsFile(false);
    setVideoUrl(localVideoUrls[activeVideo.fileName] ?? `/__local_video/${activeVideo.videoId}`);
  }, [activeVideo, localVideoUrls]);

  useEffect(() => {
    if (!activeVideo || !activeAnchor) return;
    const saved = annotations[activeAnchor.anchorId];
    const draftKey = `shuttlelab:semantic-draft:${activeAnchor.anchorId}`;
    let draft: Partial<Annotation> | null = null;
    try { draft = JSON.parse(localStorage.getItem(draftKey) || "null") as Partial<Annotation> | null; } catch { draft = null; }
    const source = saved ?? draft;
    setStrokeType(source?.stroke_type ?? "");
    const legacyOut = source?.landing_status === "observed" && source.landing_area != null && source.landing_area > 9;
    setLandingStatus(legacyOut ? "out" : source?.landing_status ?? "observed");
    setLandingArea(legacyOut ? null : source?.landing_area ?? null);
    setConfidence(source?.confidence ?? (activeAnchor.confidence === "low" ? "low" : "high"));
    setUncertainty(source?.uncertainty_frames ?? activeAnchor.uncertaintyFrames);
    setNotes(source?.notes ?? activeAnchor.notes);
    setError(""); seekFrame(activeAnchor.hitFrame);
  }, [activeAnchor, activeVideo, annotations, nextAnchor, sameRallyNext, seekFrame]);

  useEffect(() => {
    if (!activeAnchor || annotations[activeAnchor.anchorId]) return;
    const timer = window.setTimeout(() => {
      localStorage.setItem(`shuttlelab:semantic-draft:${activeAnchor.anchorId}`, JSON.stringify({
        stroke_type: strokeType, landing_status: landingStatus, landing_area: landingArea,
        confidence, uncertainty_frames: uncertainty, notes,
      }));
    }, 250);
    return () => clearTimeout(timer);
  }, [activeAnchor, annotations, confidence, landingArea, landingStatus, notes, strokeType, uncertainty]);

  const currentAnnotation = activeAnchor ? annotations[activeAnchor.anchorId] : undefined;
  const completed = useMemo(() => Object.values(annotations).filter((row) => isCompleteGold(row)).length, [annotations]);
  const selectedStroke = STROKES.find(([id]) => id === strokeType);

  const moveAnchor = useCallback((delta: number) => {
    if (!activeVideo) return; setAnchorIndex((value) => Math.max(0, Math.min(activeVideo.anchorCount - 1, value + delta)));
  }, [activeVideo]);

  const save = useCallback(async (status: "gold" | "uncertain") => {
    if (!activeVideo || !activeAnchor) return;
    if (status === "gold" && !strokeType) {
      setError("完成 Gold 需要选择球种"); return;
    }
    if (status === "gold" && landingStatus === "observed" && landingArea == null) {
      setError("场内球需要选择九宫格；出界、被接住或看不清则选择对应状态"); return;
    }
    setSync("正在保存…"); setError("");
    const payload: Annotation = {
      video_id: activeVideo.videoId, anchor_id: activeAnchor.anchorId, rally_id: activeAnchor.rallyId,
      hit_frame: activeAnchor.hitFrame, hitter: activeAnchor.hitter, stroke_type: strokeType,
      landing_kind: sameRallyNext ? "next_contact" : "terminal", landing_status: landingStatus,
      landing_area: landingStatus === "observed" ? landingArea : null,
      confidence, uncertainty_frames: uncertainty, notes, status,
    };
    try {
      const response = await fetch("/api/semantic-gold", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await response.json() as { annotation?: Annotation; error?: string };
      if (!response.ok || !data.annotation) throw new Error(data.error || "保存失败");
      setAnnotations((value) => ({ ...value, [activeAnchor.anchorId]: data.annotation! }));
      localStorage.removeItem(`shuttlelab:semantic-draft:${activeAnchor.anchorId}`);
      setSync(`已保存 revision ${data.annotation.revision}`);
      if (anchorIndex < activeVideo.anchorCount - 1) setAnchorIndex(anchorIndex + 1);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "保存失败"); setSync("保存失败"); }
  }, [activeAnchor, activeVideo, anchorIndex, confidence, landingArea, landingStatus, notes, sameRallyNext, strokeType, uncertainty]);

  const exportAll = useCallback(async () => {
    if (!catalog) return;
    setSync("正在汇总全部语义 Gold…");
    try {
      const batches = await Promise.all(catalog.videos.map(async (video) => {
        const response = await fetch(`/api/semantic-gold?video_id=${encodeURIComponent(video.videoId)}`);
        const data = await response.json() as { annotations?: Annotation[] }; return data.annotations ?? [];
      }));
      const rows = batches.flat().filter((row) => row.status === "gold").sort((a, b) => a.video_id.localeCompare(b.video_id) || a.hit_frame - b.hit_frame);
      const header = ["video_id", "rally_id", "hit_frame", "hitter", "stroke_type", "landing_kind", "landing_status", "landing_area", "landing_area_schema", "confidence", "uncertainty_frames", "notes"];
      const lines = [header.join(","), ...rows.map((row) => header.map((key) => csvCell(row[key as keyof Annotation])).join(","))];
      download("shuttlelab.semantic_gold.csv", lines.join("\r\n") + "\r\n"); setSync(`已导出 ${rows.length} 条完整 Gold`);
    } catch { setError("汇总导出失败，请先确认数据库连接"); setSync("导出失败"); }
  }, [catalog]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement | null)?.matches("input, textarea, select")) return;
      if ([" ", "ArrowLeft", "ArrowRight"].includes(event.key)) event.preventDefault();
      if (event.key === " ") { const video = videoRef.current; if (!video) return; if (video.paused) void video.play(); else video.pause(); }
      else if (event.key === "ArrowLeft") seekFrame(currentFrame - (event.shiftKey ? 10 : 1));
      else if (event.key === "ArrowRight") seekFrame(currentFrame + (event.shiftKey ? 10 : 1));
      else if (event.key.toLowerCase() === "j") moveAnchor(-1);
      else if (event.key.toLowerCase() === "k") moveAnchor(1);
      else if (event.key === "Enter") void save("gold");
    };
    addEventListener("keydown", onKey); return () => removeEventListener("keydown", onKey);
  }, [currentFrame, moveAnchor, save, seekFrame]);

  if (!catalog || !activeVideo || !activeAnchor) return <main className="semantic-loading"><strong>ShuttleLab</strong><span>{error || "正在准备语义 Gold 工作台…"}</span></main>;

  return (
    <main className="semantic-shell">
      <header className="semantic-topbar">
        <div className="semantic-brand"><b>SL</b><div><strong>ShuttleLab</strong><span>逐拍语义 Gold 工作台</span></div></div>
        <div className="semantic-blind"><i /> BLIND GOLD · 模型预测已隐藏</div>
        <div className="semantic-sync"><span>{sync}</span><button onClick={exportAll}>导出完整 Gold CSV</button></div>
      </header>

      <section className="semantic-workspace">
        <div className="semantic-main">
          <div className="semantic-context">
            <label>视频 <select value={videoIndex} onChange={(event) => setVideoIndex(Number(event.target.value))}>{catalog.videos.map((video, index) => <option value={index} key={video.videoId}>{index + 1}. {video.videoId} · {video.anchorCount}拍</option>)}</select></label>
            <span title={activeVideo.fileName}>登记文件：{activeVideo.fileName}</span>
            <div className="semantic-source-actions"><label className="primary">选择素材文件夹<input type="file" accept="video/*" multiple {...DIRECTORY_INPUT_PROPS} onChange={(event) => registerFiles(event.target.files ?? [], "已读取素材文件夹")} /></label><label>批量选择视频<input type="file" accept="video/*" multiple onChange={(event) => registerFiles(event.target.files ?? [], "已批量选择")} /></label><small>{connectedVideoCount}/{catalog.videoCount} 已连接</small></div>
            <div><b>{completed}</b><small>/ {activeVideo.anchorCount} 本段完成</small></div>
          </div>

          <div className="semantic-video-card">
            <video ref={videoRef} src={videoUrl ?? undefined} preload="auto" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
              onTimeUpdate={(event) => setCurrentFrame(Math.round(event.currentTarget.currentTime * activeVideo.fps))}
              onError={() => { if (videoUrl?.startsWith("/__local_video/")) { setVideoNeedsFile(true); setVideoUrl(null); } }} />
            {!videoUrl && <div className="semantic-video-empty"><strong>选择一次素材文件夹，之后自动切换视频</strong><span>当前登记文件：{activeVideo.fileName}</span><span>{sourceStatus}；视频只在本机读取，不会上传。</span><div><label className="primary">选择整个素材文件夹<input type="file" accept="video/*" multiple {...DIRECTORY_INPUT_PROPS} onChange={(event) => registerFiles(event.target.files ?? [], "已读取素材文件夹")} /></label><label>只选当前视频<input type="file" accept="video/*" onChange={(event) => { const file = event.target.files?.[0]; if (file) registerFiles([file], "已选择当前视频"); }} /></label></div></div>}
            <div className="semantic-frame"><span>HIT ANCHOR</span><b>F{activeAnchor.hitFrame}</b><em>±{activeAnchor.uncertaintyFrames}帧</em></div>
            <div className="semantic-hitter">{activeAnchor.rallyId} · {activeAnchor.hitter === "upper" ? "远端击球" : activeAnchor.hitter === "lower" ? "近端击球" : "击球方不确定"}</div>
          </div>

          <div className="semantic-transport">
            <span>当前 F{currentFrame} · {(currentFrame / activeVideo.fps).toFixed(3)}s</span>
            <div><button onClick={() => seekFrame(currentFrame - 10)}>−10</button><button onClick={() => seekFrame(currentFrame - 1)}>−1</button><button className="play" onClick={() => { const v = videoRef.current; if (!v) return; if (v.paused) void v.play(); else v.pause(); }}>{playing ? "Ⅱ" : "▶"}</button><button onClick={() => seekFrame(currentFrame + 1)}>+1</button><button onClick={() => seekFrame(currentFrame + 10)}>+10</button></div>
            <button onClick={() => seekFrame(activeAnchor.hitFrame)}>回到击球帧</button>
          </div>

          <section className="semantic-form">
            <div className="semantic-section-title"><div><span>01</span><strong>选择球种</strong></div><small>必须依据视频独立判断；悬停可看定义</small></div>
            <div className="semantic-strokes">{STROKES.map(([id, zh, help], index) => <button key={id} title={help} className={strokeType === id ? "selected" : ""} onClick={() => setStrokeType(id)}><em>{String(index + 1).padStart(2, "0")}</em><strong>{zh}</strong><span>{id}</span></button>)}</div>
            <div className="semantic-definition"><b>{selectedStroke?.[1] ?? "尚未选择球种"}</b><span>{selectedStroke?.[2] ?? "选择后这里会显示统一判定说明；不要根据模型猜测。"}</span></div>

            <div className="semantic-section-title landing"><div><span>02</span><strong>选择目的地区域</strong></div><small>{sameRallyNext ? "普通拍：对手下一次触球区域" : "回合末拍：选择终止区域或特殊状态"}</small></div>
            <div className="semantic-area-workbench">
              <div className="semantic-area-picker">
                <div className={`semantic-area-court ${landingStatus !== "observed" ? "muted" : ""}`}><span className="area-net">球网</span>{COURT_AREAS.map(([id, label]) => <button key={id} className={landingStatus === "observed" && landingArea === id ? "selected" : ""} onClick={() => { setLandingStatus("observed"); setLandingArea(id); }}><em>{id}</em><strong>{label}</strong></button>)}<small>点击任一区域即切换为场内 · 前=靠近球网</small></div>
                <div className="semantic-area-specials">
                  <header><strong>特殊状态</strong><span>只在不属于场内九宫格时选择</span></header>
                  <div>
                    <button className={landingStatus === "out" ? "selected" : ""} onClick={() => { setLandingStatus("out"); setLandingArea(null); }}><strong>出界</strong><span>不细分方向</span></button>
                    <button className={landingStatus === "caught" ? "selected" : ""} onClick={() => { setLandingStatus("caught"); setLandingArea(null); }}><strong>被接住</strong><span>没有实际落点</span></button>
                    <button className={landingStatus === "unobservable" ? "selected" : ""} onClick={() => { setLandingStatus("unobservable"); setLandingArea(null); }}><strong>看不清</strong><span>不猜测区域</span></button>
                  </div>
                  <footer><b>{landingStatus === "observed" ? landingArea == null ? "等待选择场内区域" : `已选择场内 ${landingArea} 区` : landingStatus === "out" ? "已标记：出界" : landingStatus === "caught" ? "已标记：被接住" : "已标记：看不清"}</b><span>{landingStatus === "observed" ? "landing_area 保存为 1–9" : "landing_area 明确留空"}</span></footer>
                </div>
              </div>
            </div>

            <div className="semantic-details"><label>置信度<select value={confidence} onChange={(event) => setConfidence(event.target.value as typeof confidence)}><option value="high">高</option><option value="medium">中</option><option value="low">低</option></select></label><label>帧误差<select value={uncertainty} onChange={(event) => setUncertainty(Number(event.target.value))}>{[0,1,2,3,4,5,8,10].map((value) => <option key={value} value={value}>±{value}帧</option>)}</select></label><label className="notes">备注<input value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="遮挡、擦网、边界球等" /></label></div>
            {error && <div className="semantic-error">{error}</div>}
            <div className="semantic-save"><button className="uncertain" onClick={() => void save("uncertain")}>无法可靠判断</button><button className="primary" onClick={() => void save("gold")}>{currentAnnotation ? "保存新修订并下一拍" : "保存 Gold 并下一拍"}<kbd>Enter</kbd></button></div>
          </section>
        </div>

        <aside className="semantic-rail">
          <div className="semantic-rail-head"><div><span>ANNOTATION QUEUE</span><strong>{activeVideo.videoId}</strong></div><b>{anchorIndex + 1}/{activeVideo.anchorCount}</b></div>
          <div className="semantic-progress"><div><i style={{ width: `${Math.round((completed / activeVideo.anchorCount) * 100)}%` }} /></div><span>{Math.round((completed / activeVideo.anchorCount) * 100)}% 完成</span></div>
          <div className="semantic-queue">{activeVideo.anchors.map((anchor, index) => { const value = annotations[anchor.anchorId]; const complete = isCompleteGold(value); const queueStatus = complete ? "gold" : value?.status === "uncertain" ? "uncertain" : "pending"; return <button key={anchor.anchorId} className={`${index === anchorIndex ? "active" : ""} ${queueStatus}`} onClick={() => setAnchorIndex(index)}><em>{index + 1}</em><div><strong>{anchor.rallyId}</strong><span>F{anchor.hitFrame} · {anchor.hitter}</span></div><b>{complete ? "✓" : value?.status === "uncertain" ? "?" : "·"}</b></button>; })}</div>
          <div className="semantic-rail-nav"><button onClick={() => moveAnchor(-1)} disabled={anchorIndex === 0}>上一拍 <kbd>J</kbd></button><button onClick={() => moveAnchor(1)} disabled={anchorIndex === activeVideo.anchorCount - 1}>下一拍 <kbd>K</kbd></button></div>
          <div className="semantic-protocol"><strong>Gold规则</strong><p>只标已有人工击球锚点；不看模型预测。场内球选九宫格，出界只选“出界”，不再细分方向。球被接住就选“被接住”，看不清选“看不清”。</p>{videoNeedsFile && <span>找不到登记文件：{activeVideo.fileName}。点击“选择素材文件夹”，浏览器会一次读取其中所有视频并自动匹配。</span>}</div>
        </aside>
      </section>
    </main>
  );
}
