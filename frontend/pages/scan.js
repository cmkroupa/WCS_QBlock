import Link from 'next/link';
import Script from 'next/script';
import { useEffect, useRef, useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5001';

function pct(score) {
  return Math.round(Math.max(0, Math.min(1, Number(score) || 0)) * 100);
}

export default function ScanPage() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef    = useRef(null);

  const [status, setStatus]           = useState('booting');
  const [detectedUrl, setDetectedUrl] = useState('');
  const [verdict, setVerdict]         = useState(null);
  const [error, setError]             = useState('');

  const stopCamera = () => {
    if (rafRef.current)    { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current)  videoRef.current.srcObject = null;
  };

  const processUrl = async (url) => {
    setDetectedUrl(url);
    setStatus('loading');
    stopCamera();
    try {
      const res  = await fetch(`${API_BASE}/api/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Backend request failed');

      setVerdict({
        prediction: data.blend?.final_prediction ?? 'unknown',
        score:      data.blend?.final_score ?? 0,
        url:        data.url || url,
      });
      setStatus('result');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  const scanFrame = () => {
    if (status !== 'scanning') return;
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) {
      rafRef.current = requestAnimationFrame(scanFrame);
      return;
    }
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (window.jsQR) {
      const code = window.jsQR(imageData.data, imageData.width, imageData.height);
      if (code?.data) { processUrl(code.data); return; }
    }
    rafRef.current = requestAnimationFrame(scanFrame);
  };

  const startCameraAndScan = async () => {
    setError(''); setVerdict(null); setDetectedUrl('');
    setStatus('booting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } }, audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      setStatus('scanning');
    } catch (err) {
      setError(`Camera access failed: ${err.message}`);
      setStatus('error');
    }
  };

  useEffect(() => {
    if (status === 'scanning') rafRef.current = requestAnimationFrame(scanFrame);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [status]);

  useEffect(() => { startCameraAndScan(); return () => stopCamera(); }, []);

  const isSafe       = verdict?.prediction === 'safe';
  const isSuspicious = verdict?.prediction === 'suspicious';

  return (
    <>
      <Script src="https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.js" strategy="afterInteractive" />

      <main className="scanPage">
        <video ref={videoRef} playsInline muted className="cameraFeed" />
        <canvas ref={canvasRef} hidden />

        {/* ── Booting ── */}
        {status === 'booting' && (
          <div className="overlay booting">
            <div className="bootRing" />
            <p>Starting camera…</p>
          </div>
        )}

        {/* ── Scanning ── */}
        {status === 'scanning' && (
          <div className="overlay scanning">
            <div className="hudTop">
              <Link href="/" className="backLink">← Back</Link>
              <span>LIVE QR SCAN</span>
            </div>
            <div className="scanFrame" />
            <p className="hint">Point camera at a QR code</p>
          </div>
        )}

        {/* ── Loading ── */}
        {status === 'loading' && (
          <div className="overlay loading">
            <div className="spinner" />
            <p className="loadingUrl">{detectedUrl}</p>
            <p className="loadingLabel">Checking…</p>
          </div>
        )}

        {/* ── Result ── */}
        {status === 'result' && verdict && (
          <div className="overlay result">
            <div className={`blockedCard ${isSafe ? 'blockedCardSafe' : isSuspicious ? 'blockedCardSuspicious' : 'blockedCardPhishing'}`}>
              <p className={`blockedLabel ${isSafe ? 'blockedLabelSafe' : 'blockedLabelDanger'}`}>
                {isSafe ? 'Safe' : isSuspicious ? 'Suspected Phishing' : 'Phishing Detected'}
              </p>
              <p className="blockedConf">{pct(verdict.score)}% phishing probability</p>
              <p className="blockedUrl">{verdict.url}</p>
              <button className="scanAgainBtn" onClick={startCameraAndScan}>Scan Again</button>
            </div>
          </div>
        )}

        {/* ── Error ── */}
        {status === 'error' && (
          <div className="overlay error">
            <p className="errorTitle">Scan failed</p>
            <p className="errorMsg">{error || 'An unknown error occurred.'}</p>
            <div className="actionRow">
              <button onClick={startCameraAndScan}>Retry</button>
              <Link href="/" className="ghostBtn">Home</Link>
            </div>
          </div>
        )}
      </main>
    </>
  );
}
