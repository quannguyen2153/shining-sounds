"use client";

import React, { useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000/audio/separate/stream";

const AudioTool: React.FC = () => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [drumsUrl, setDrumsUrl] = useState<string>("");
  const [bassUrl, setBassUrl] = useState<string>("");
  const [vocalsUrl, setVocalsUrl] = useState<string>("");
  const [otherUrl, setOtherUrl] = useState<string>("");
  const [drumsFile, setDrumsFile] = useState<File | null>(null);
  const [bassFile, setBassFile] = useState<File | null>(null);
  const [vocalsFile, setVocalsFile] = useState<File | null>(null);
  const [otherFile, setOtherFile] = useState<File | null>(null);
  const [separating, setSeparating] = useState(false);
  const [separationStatus, setSeparationStatus] = useState<string>("");
  const [progress, setProgress] = useState({ processed: 0, total: 0, eta: "--:--:--" });
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle YouTube fetch
  const handleFetchYoutube = async () => {
    if (!youtubeUrl) return;
    setLoading(true);
    try {
      const res = await fetch(
        `/api/audio/youtube?url=${encodeURIComponent(youtubeUrl)}`
      );
      if (!res.ok) throw new Error("Failed to fetch audio");
      const contentType = res.headers.get("Content-Type") || "audio/mpeg";
      const blob = await res.blob();
      const ext = contentType.split("/")[1]?.split(";")[0] || "mp3";
      const file = new File([blob], `youtube-audio.${ext}`, {
        type: contentType,
      });
      setAudioFile(file);
      setAudioUrl(URL.createObjectURL(file));
    } catch (e) {
      alert("Error fetching audio");
    } finally {
      setLoading(false);
    }
  };

  // Handle file upload
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAudioFile(file);
      setAudioUrl(URL.createObjectURL(file));
    }
  };

  // Placeholder for separation
  const handleSeparate = async () => {
    if (!audioFile) return alert("No audio file selected");
    setDrumsUrl("");
    setBassUrl("");
    setVocalsUrl("");
    setOtherUrl("");
    setDrumsFile(null);
    setBassFile(null);
    setVocalsFile(null);
    setOtherFile(null);
    setSeparationStatus("");
    setSeparating(true);
    setProgress({ processed: 0, total: 0, eta: "--:--:--" });
    try {
      const formData = new FormData();
      formData.append("file", audioFile);
      formData.append("output_format", "mp3");
      const response = await fetch(API_URL, { method: "POST", body: formData });
      if (!response.body || !response.ok) {
        setSeparationStatus("Separation failed.");
        setSeparating(false);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "",
        currentEvent = null,
        currentData = "";
      const parseLine = (line: string) =>
        line.startsWith("event:")
          ? { event: line.slice(6).trim() }
          : line.startsWith("data:")
          ? { data: line.slice(5).trim() }
          : {};
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop()!;
        for (let line of lines) {
          line = line.trim();
          if (line === "") {
            if (currentEvent === "status") {
              setSeparationStatus(currentData);
            } else if (currentEvent === "progress") {
              try {
                const data = JSON.parse(currentData);
                setProgress({
                  processed: data.processed,
                  total: data.total,
                  eta: data.eta,
                });
              } catch {}
            } else if (currentEvent === "stem") {
              try {
                const meta = JSON.parse(currentData);
                // meta.name: "drums", "bass", "vocals", "other"
                // meta.id: id for audio
                const url = `http://127.0.0.1:8000/audio/get/${meta.id}`;
                // Fetch the file as blob, create File, and set URL
                fetch(url).then(async (res) => {
                  const contentType =
                    res.headers.get("Content-Type") || "audio/mpeg";
                  const ext = contentType.split("/")[1]?.split(";")[0] || "mp3";
                  const blob = await res.blob();
                  const file = new File([blob], `${meta.name}.${ext}`, {
                    type: contentType,
                  });
                  const objUrl = URL.createObjectURL(file);
                  if (meta.name === "drums.mp3") {
                    setDrumsFile(file);
                    setDrumsUrl(objUrl);
                  } else if (meta.name === "bass.mp3") {
                    setBassFile(file);
                    setBassUrl(objUrl);
                  } else if (meta.name === "vocals.mp3") {
                    setVocalsFile(file);
                    setVocalsUrl(objUrl);
                  } else if (meta.name === "other.mp3") {
                    setOtherFile(file);
                    setOtherUrl(objUrl);
                  }
                });
              } catch {}
            } else if (currentEvent === "done") {
              setSeparationStatus("Done!");
            }
            currentEvent = null;
            currentData = "";
            continue;
          }
          const parsed = parseLine(line);
          if ((parsed as any).event) currentEvent = (parsed as any).event;
          else if ((parsed as any).data !== undefined)
            currentData += (parsed as any).data;
        }
      }
    } catch (e) {
      setSeparationStatus("Separation failed.");
    } finally {
      setSeparating(false);
    }
  };

  // Validate YouTube URL (simple check)
  const isValidYoutubeUrl = (url: string) => {
    return /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\/.+/.test(url);
  };

  // Auto-fetch when valid YouTube URL is entered
  React.useEffect(() => {
    if (isValidYoutubeUrl(youtubeUrl)) {
      handleFetchYoutube();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [youtubeUrl]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Row 1: Input */}
      <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
        <input
          type="text"
          placeholder="Paste YouTube URL here..."
          value={youtubeUrl}
          onChange={(e) => setYoutubeUrl(e.target.value)}
          style={{
            flex: 1,
            padding: "10px 14px",
            borderRadius: 8,
            border: "1px solid #ccc",
            fontSize: 16,
            outline: "none",
            transition: "border 0.2s",
            boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
            marginRight: 8,
            background: loading ? "#f5f5f5" : undefined,
            color: loading ? "#aaa" : undefined,
          }}
          disabled={loading}
        />
        {/* Removed Fetch button */}
        <input
          type="file"
          accept="audio/*"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          style={{
            padding: "10px 18px",
            borderRadius: 8,
            border: "none",
            background: "#0070f3",
            color: "#fff",
            fontWeight: 600,
            fontSize: 16,
            cursor: "pointer",
            boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
            marginLeft: 8,
            transition: "background 0.2s",
          }}
        >
          Upload Audio
        </button>
      </div>
      {/* Row 2: Playback */}
      <div>
        {audioUrl ? (
          <audio controls src={audioUrl} style={{ width: "100%" }} />
        ) : (
          <span>
            {loading ? "Fetching audio from YouTube..." : "No audio selected"}
          </span>
        )}
      </div>
      {/* Row 3: Separate */}
      <div>
        <button
          onClick={handleSeparate}
          disabled={!audioFile || separating}
          style={{
            padding: "12px 22px",
            borderRadius: 8,
            border: "none",
            background: !audioFile || separating ? "#ccc" : "#10b981",
            color: "#fff",
            fontWeight: 600,
            fontSize: 17,
            cursor: !audioFile || separating ? "not-allowed" : "pointer",
            boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
            marginTop: 8,
            transition: "background 0.2s",
          }}
        >
          {separating ? "Separating..." : "Separate Audio Sources"}
        </button>
        {separationStatus && (
          <div
            style={{
              marginTop: 12,
              color: separationStatus === "Done!" ? "#10b981" : "#333",
            }}
          >
            {separationStatus}
          </div>
        )}
        {progress.total > 0 && (
          <div style={{ marginTop: 12 }}>
            <p>
              Segment Progress: <span>{progress.processed}/{progress.total}</span> | ETA: <span>{progress.eta}</span>
            </p>
            <progress
              value={progress.processed}
              max={progress.total}
              style={{ width: "100%" }}
            />
          </div>
        )}
      </div>
      {/* Row 4: Stems Playback */}
      {(drumsUrl || bassUrl || vocalsUrl || otherUrl) && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 16,
            marginTop: 24,
          }}
        >
          {drumsUrl && (
            <div>
              <strong>Drums</strong>
              <audio
                controls
                src={drumsUrl}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          )}
          {bassUrl && (
            <div>
              <strong>Bass</strong>
              <audio
                controls
                src={bassUrl}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          )}
          {vocalsUrl && (
            <div>
              <strong>Vocals</strong>
              <audio
                controls
                src={vocalsUrl}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          )}
          {otherUrl && (
            <div>
              <strong>Other</strong>
              <audio
                controls
                src={otherUrl}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AudioTool;
